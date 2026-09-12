from __future__ import annotations

from datetime import timedelta
import math
from typing import Any

import pandas as pd

STATUS = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
METHOD = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}


class OutputVerifier:
    """Final deterministic guardrail for challenge output."""

    def __init__(self, data: dict[str, pd.DataFrame]):
        self.data = data

    def verify_row(self, request: pd.Series, row: dict[str, Any]) -> list[str]:
        errors = []
        amount = float(row["amount_safe_to_pay"])
        requested = float(request.requested_amount)
        if not (0 <= amount <= requested + 1e-6):
            errors.append("amount_safe_to_pay_out_of_range")
        status = row["affordability_status"]
        method = row["recommended_payment_method"]
        if status not in STATUS:
            errors.append("invalid_status")
        if method not in METHOD:
            errors.append("invalid_method")
        if method == "not_recommended" and row["payment_plan"] != "none":
            errors.append("decline_must_have_none_plan")
        if status == "affordable_now":
            earliest = str(row.get("earliest_date_for_full_payment", ""))
            if earliest != str(request.request_date):
                errors.append("affordable_now_requires_request_date")
        if status == "affordable_later" and method != "wait":
            errors.append("affordable_later_requires_wait")
        if method == "wait" and status != "affordable_later":
            errors.append("wait_requires_affordable_later")
        if method == "partial_payment":
            if status != "affordable_with_plan":
                errors.append("partial_requires_plan_status")
            if not bool(request.allows_partial_payment):
                errors.append("partial_not_allowed")
            if not (0 < amount < requested):
                errors.append("partial_requires_strictly_partial_amount")
            parts = str(row["payment_plan"]).split("|")
            if len(parts) != 2:
                errors.append("partial_requires_two_payments")
            else:
                try:
                    dates = [x.split(":", 1)[0] for x in parts]
                    vals = [float(x.split(":", 1)[1]) for x in parts]
                    if dates[0] != str(request.request_date):
                        errors.append("partial_first_payment_must_be_request_date")
                    if abs(sum(vals) - requested) > 0.02:
                        errors.append("partial_total_mismatch")
                    if abs(vals[0] - amount) > 0.02:
                        errors.append("partial_first_amount_mismatch")
                    if vals[1] < -1e-9 or dates[1] > str(request.desired_completion_date):
                        errors.append("partial_remaining_payment_invalid")
                except Exception:
                    errors.append("invalid_partial_plan")
        if method == "installments":
            if status != "affordable_with_plan":
                errors.append("installments_requires_plan_status")
            if not self._matches_option(request.request_id, str(row["payment_plan"])):
                errors.append("installment_plan_not_in_options")
        if method == "full_payment":
            if str(row["payment_plan"]) == "none":
                errors.append("full_payment_requires_plan")
        if method == "wait":
            try:
                parts = str(row["payment_plan"]).split(":", 1)
                if len(parts) != 2 or abs(float(parts[1]) - requested) > 0.02:
                    errors.append("wait_plan_must_be_full_request")
            except Exception:
                errors.append("invalid_wait_plan")
        changes = str(row["spending_changes_needed"])
        if changes != "none":
            change_parts = changes.split("|")
            if len(change_parts) > 3:
                errors.append("too_many_spending_changes")
            ids = []
            for part in change_parts:
                if part.startswith("stop:"):
                    ids.append(part.split(":", 1)[1])
                elif part.startswith("reduce_to:"):
                    bits = part.split(":")
                    if len(bits) != 3:
                        errors.append("invalid_reduce_change")
                    else:
                        ids.append(bits[1])
                        try:
                            if float(bits[2]) < 0:
                                errors.append("negative_reduction")
                        except ValueError:
                            errors.append("invalid_reduce_amount")
                else:
                    errors.append("invalid_spending_change")
            if len(ids) != len(set(ids)):
                errors.append("duplicate_spending_change_event")
        return errors

    def _matches_option(self, request_id: str, plan: str) -> bool:
        rows = self.data["payment_options"][self.data["payment_options"].request_id.eq(request_id)]
        for _, opt in rows.iterrows():
            try:
                if any(pd.isna(opt.get(k)) for k in ("first_payment_date", "payment_frequency_days", "number_of_payments", "payment_amount")):
                    continue
                first = pd.Timestamp(opt.first_payment_date)
                freq = int(opt.payment_frequency_days)
                n = int(opt.number_of_payments)
                amount = float(opt.payment_amount)
                if n <= 0 or freq < 0 or not math.isfinite(amount):
                    continue
                expected = "|".join(f"{(first + timedelta(days=i*freq)):%Y-%m-%d}:{amount:.2f}" for i in range(n))
                if expected == plan:
                    return True
            except (TypeError, ValueError, OverflowError):
                continue
        return False

    def verify(self, requests: pd.DataFrame, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        clean, issues = [], []
        for req, row in zip(requests.itertuples(index=False), rows):
            req_series = pd.Series(req._asdict())
            errors = self.verify_row(req_series, row)
            if errors:
                issues.append({"request_id": req.request_id, "errors": errors})
            clean.append({k: row[k] for k in ["request_id", "amount_safe_to_pay", "affordability_status", "recommended_payment_method", "payment_plan", "earliest_date_for_full_payment", "spending_changes_needed", "decision_explanation"]})
        return clean, issues
