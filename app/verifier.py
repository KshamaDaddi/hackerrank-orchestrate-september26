from __future__ import annotations

from datetime import timedelta
import math
import re
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
        if not (0 <= amount <= requested + 1e-6): errors.append("amount_safe_to_pay_out_of_range")
        if row["affordability_status"] not in STATUS: errors.append("invalid_status")
        if row["recommended_payment_method"] not in METHOD: errors.append("invalid_method")
        if row["recommended_payment_method"] == "not_recommended" and row["payment_plan"] != "none": errors.append("decline_must_have_none_plan")
        if row["recommended_payment_method"] == "partial_payment":
            if not bool(request.allows_partial_payment): errors.append("partial_not_allowed")
            parts = str(row["payment_plan"]).split("|")
            if len(parts) != 2: errors.append("partial_requires_two_payments")
            else:
                try:
                    vals = [float(x.split(":", 1)[1]) for x in parts]
                    if abs(sum(vals) - requested) > 0.02: errors.append("partial_total_mismatch")
                except Exception: errors.append("invalid_partial_plan")
        if row["recommended_payment_method"] == "installments":
            if not self._matches_option(request.request_id, str(row["payment_plan"])):
                errors.append("installment_plan_not_in_options")
        if row["spending_changes_needed"] != "none" and len(str(row["spending_changes_needed"]).split("|")) > 3:
            errors.append("too_many_spending_changes")
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
                if expected == plan: return True
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
