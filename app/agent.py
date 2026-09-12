from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from app.ai_layer import AIInterpreter
from app.data_loader import DataLoader
from app.financial_engine import FinancialEngine, Payment
from app.risk_model import RiskModel


class FinancialAgent:
    """Hybrid agent: AI interpretation + deterministic plan search + deterministic verification."""

    def __init__(self, dataset_dir: str | Path = "dataset"):
        self.loader = DataLoader(dataset_dir)
        self.data = self.loader.load_all()
        self.engine = FinancialEngine(self.data)
        self.ai = AIInterpreter()
        self.risk = RiskModel()
        self.risk.fit(self.data["profiles"])

    @staticmethod
    def _money(v: float) -> float:
        return round(float(v) + 1e-9, 2)

    @staticmethod
    def _method_map(value: str) -> str:
        s = str(value).lower().strip()
        if "install" in s: return "installments"
        if "partial" in s: return "partial_payment"
        if "full" in s: return "full_payment"
        return s

    def _option_candidates(self, request: pd.Series, profile: pd.Series) -> list[dict[str, Any]]:
        accepted = self.engine.accepted_methods(profile)
        rows = self.data["payment_options"][self.data["payment_options"].request_id.eq(request.request_id)].copy()
        candidates = []
        for _, opt in rows.iterrows():
            method = self._method_map(opt.payment_method)
            if method not in accepted:
                continue
            payments = self.engine.option_payments(opt)
            # Respect completion deadline; every payment must be completed by it.
            if payments and payments[-1].date > pd.Timestamp(request.desired_completion_date):
                continue
            sim = self.engine.simulate(request.user_id, request.request_date, payments)
            candidates.append({"kind": method, "option": opt, "payments": payments, "sim": sim, "fee": float(opt.financing_fee), "total": float(opt.total_payable_amount)})
        return candidates

    def _full_candidate(self, request: pd.Series, amount: float) -> dict[str, Any]:
        pay = Payment(pd.Timestamp(request.request_date), amount, "full")
        return {"kind": "full_payment", "option": None, "payments": [pay], "sim": self.engine.simulate(request.user_id, request.request_date, [pay]), "fee": 0.0, "total": amount}

    def _partial_candidate(self, request: pd.Series, amount_today: float, date_full: pd.Timestamp | None) -> dict[str, Any] | None:
        if not bool(request.allows_partial_payment) or amount_today <= 1e-8 or amount_today >= float(request.requested_amount) - 1e-8 or date_full is None:
            return None
        if date_full > pd.Timestamp(request.desired_completion_date):
            return None
        payments = [Payment(pd.Timestamp(request.request_date), amount_today, "partial"), Payment(date_full, float(request.requested_amount) - amount_today, "partial")]
        sim = self.engine.simulate(request.user_id, request.request_date, payments)
        return {"kind": "partial_payment", "option": None, "payments": payments, "sim": sim, "fee": 0.0, "total": float(request.requested_amount)} if sim.safe else None

    def _change_candidate(self, request: pd.Series, profile: pd.Series, base: dict[str, Any]) -> dict[str, Any] | None:
        changes = self.engine.flexible_changes(request.user_id, request.request_date, float(request.requested_amount), request.desired_completion_date)
        if not changes: return None
        # Try stopping one recurring flexible event at a time; only accept a change if it makes a complete plan safe.
        for change in changes:
            test = dict(change)
            test["new_amount"] = 0.0
            sim = self.engine.adjusted_simulation(request.user_id, request.request_date, base["payments"], [test])
            if sim.safe:
                return {**base, "sim": sim, "changes": [test], "change_count": 1}
        return None

    def _rank(self, candidate: dict[str, Any], request: pd.Series) -> tuple:
        payments = candidate["payments"]
        completes = bool(payments) and payments[-1].date <= pd.Timestamp(request.desired_completion_date)
        changes = len(candidate.get("changes", []))
        start = payments[0].date if payments else pd.Timestamp.max
        count = len(payments)
        option_id = str(candidate["option"].payment_option_id) if candidate.get("option") is not None else "zzzz"
        return (0 if completes else 1, changes, candidate["total"], start, count, option_id)

    def decide(self, request: pd.Series) -> dict[str, Any]:
        profile = self.engine.profile(request.user_id)
        requested = float(request.requested_amount)
        today_safe = self.engine.max_safe_today(request.user_id, request.request_date, requested)
        earliest = self.engine.earliest_full_payment(request.user_id, request.request_date, requested, request.desired_completion_date)
        interpretation = self.ai.interpret(str(request.request_text))
        risk = self.risk.score(profile)

        candidates: list[dict[str, Any]] = []
        full = self._full_candidate(request, requested)
        if full["sim"].safe and "full_payment" in self.engine.accepted_methods(profile):
            candidates.append(full)
        candidates.extend(x for x in self._option_candidates(request, profile) if x["sim"].safe)
        partial = self._partial_candidate(request, today_safe, earliest)
        if partial and "partial_payment" in self.engine.accepted_methods(profile):
            candidates.append(partial)
        # Spending changes are only considered as a last resort after no-change plans.
        if not candidates:
            base = full if "full_payment" in self.engine.accepted_methods(profile) else None
            if base:
                changed = self._change_candidate(request, profile, base)
                if changed: candidates.append(changed)

        if candidates:
            best = min(candidates, key=lambda x: self._rank(x, request))
            method = best["kind"]
            plan = "|".join(f"{p.date:%Y-%m-%d}:{self._money(p.amount):.2f}" for p in best["payments"])
            status = "affordable_now" if method == "full_payment" and best["payments"][0].date == pd.Timestamp(request.request_date) else "affordable_with_plan"
            if method == "full_payment" and best.get("option") is None and earliest is not None and earliest > pd.Timestamp(request.request_date):
                status = "affordable_later"
                method = "wait"
            changes = best.get("changes", [])
            change_text = "none" if not changes else "|".join(f"stop:{c['event_id']}" for c in changes)
            cur = str(profile.home_currency)
            reason = f"{method.replace('_',' ')} is safe in {cur}. The projected minimum balance is {self._money(best['sim'].minimum_balance):.2f} versus the required {self._money(profile.minimum_balance_to_keep):.2f}."
            if changes: reason += " A flexible recurring expense is stopped to preserve the safety buffer."
            return {
                "request_id": request.request_id,
                "amount_safe_to_pay": self._money(today_safe),
                "affordability_status": status,
                "recommended_payment_method": method,
                "payment_plan": plan,
                "earliest_date_for_full_payment": "" if earliest is None else f"{earliest:%Y-%m-%d}",
                "spending_changes_needed": change_text,
                "decision_explanation": reason,
                "_risk_score": risk,
                "_ai_intent": interpretation.get("intent", "other"),
            }

        # No safe immediate plan. If the full amount becomes safe later, wait is valid only when full payment is accepted.
        if earliest is not None and "full_payment" in self.engine.accepted_methods(profile):
            sim = self.engine.simulate(request.user_id, request.request_date, [Payment(earliest, requested, "wait")])
            return {
                "request_id": request.request_id,
                "amount_safe_to_pay": self._money(today_safe),
                "affordability_status": "affordable_later",
                "recommended_payment_method": "wait",
                "payment_plan": f"{earliest:%Y-%m-%d}:{requested:.2f}",
                "earliest_date_for_full_payment": f"{earliest:%Y-%m-%d}",
                "spending_changes_needed": "none",
                "decision_explanation": f"Wait until {earliest:%Y-%m-%d}; paying earlier would breach the minimum balance constraint.",
                "_risk_score": risk,
                "_ai_intent": interpretation.get("intent", "other"),
            }
        return {
            "request_id": request.request_id,
            "amount_safe_to_pay": self._money(today_safe),
            "affordability_status": "not_affordable",
            "recommended_payment_method": "not_recommended",
            "payment_plan": "none",
            "earliest_date_for_full_payment": "" if earliest is None else f"{earliest:%Y-%m-%d}",
            "spending_changes_needed": "none",
            "decision_explanation": f"Do not proceed. No eligible plan completes the full request safely while maintaining the required minimum balance.",
            "_risk_score": risk,
            "_ai_intent": interpretation.get("intent", "other"),
        }
