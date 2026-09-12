from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

from app.ai_layer import AIInterpreter
from app.data_loader import DataLoader
from app.financial_engine import FinancialEngine, Payment
from app.multi_agent import MultiAgentReasoner
from app.risk_model import RiskModel


class FinancialAgent:
    """Production-style hybrid agent: specialized LLMs propose semantic insights; code verifies money."""

    def __init__(self, dataset_dir: str | Path = "dataset"):
        self.loader = DataLoader(dataset_dir)
        self.data = self.loader.load_all()
        self.engine = FinancialEngine(self.data, dataset_dir=str(dataset_dir))
        self.ai = AIInterpreter()
        self.reasoner = MultiAgentReasoner()
        self.risk = RiskModel()
        self.risk.fit(self.data["profiles"])
        self.audit_traces: dict[str, dict[str, Any]] = {}

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
        rows = self.data["payment_options"][self.data["payment_options"].request_id.eq(request.request_id)]
        candidates = []
        for _, opt in rows.iterrows():
            method = self._method_map(opt.payment_method)
            if method not in accepted: continue
            payments = self.engine.option_payments(opt)
            if not payments or payments[-1].date > pd.Timestamp(request.desired_completion_date): continue
            sim = self.engine.simulate(request.user_id, request.request_date, payments)
            if sim.safe:
                candidates.append({"kind": method, "option": opt, "payments": payments, "sim": sim,
                                   "fee": float(opt.financing_fee), "total": float(opt.total_payable_amount), "changes": []})
        return candidates

    def _full_candidate(self, request: pd.Series, amount: float) -> dict[str, Any]:
        pay = Payment(pd.Timestamp(request.request_date), amount, "full")
        return {"kind": "full_payment", "option": None, "payments": [pay],
                "sim": self.engine.simulate(request.user_id, request.request_date, [pay]),
                "fee": 0.0, "total": amount, "changes": []}

    def _partial_candidate(self, request: pd.Series, amount_today: float, date_full: pd.Timestamp | None) -> dict[str, Any] | None:
        if not bool(request.allows_partial_payment) or amount_today <= 1e-8 or amount_today >= float(request.requested_amount) - 1e-8 or date_full is None:
            return None
        if date_full > pd.Timestamp(request.desired_completion_date): return None
        payments = [Payment(pd.Timestamp(request.request_date), amount_today, "partial"), Payment(date_full, float(request.requested_amount) - amount_today, "partial")]
        sim = self.engine.simulate(request.user_id, request.request_date, payments)
        if not sim.safe: return None
        return {"kind": "partial_payment", "option": None, "payments": payments, "sim": sim,
                "fee": 0.0, "total": float(request.requested_amount), "changes": []}

    def _change_candidates(self, request: pd.Series, base: dict[str, Any]) -> list[dict[str, Any]]:
        changes = self.engine.flexible_changes(request.user_id, request.request_date, float(request.requested_amount), request.desired_completion_date)
        results = []
        for n in range(1, min(3, len(changes)) + 1):
            for combo in combinations(changes, n):
                if len({str(c["event_id"]) for c in combo}) != n: continue
                sim = self.engine.adjusted_simulation(request.user_id, request.request_date, base["payments"], list(combo))
                if sim.safe:
                    results.append({**base, "sim": sim, "changes": list(combo), "change_count": n})
        return results

    def _rank(self, candidate: dict[str, Any], request: pd.Series) -> tuple:
        payments = candidate["payments"]
        completes = bool(payments) and payments[-1].date <= pd.Timestamp(request.desired_completion_date)
        option_id = str(candidate["option"].payment_option_id) if candidate.get("option") is not None else "zzzz"
        return (0 if completes else 1, len(candidate.get("changes", [])), candidate["total"],
                payments[0].date if payments else pd.Timestamp.max, len(payments), option_id)

    def decide(self, request: pd.Series) -> dict[str, Any]:
        profile = self.engine.profile(request.user_id)
        requested = float(request.requested_amount)
        intent = self.reasoner.understand_request(request.to_dict())
        today_safe = self.engine.max_safe_today(request.user_id, request.request_date, requested)
        earliest = self.engine.earliest_full_payment(request.user_id, request.request_date, requested, request.desired_completion_date)
        risk = self.risk.score(profile)
        candidates: list[dict[str, Any]] = []

        full = self._full_candidate(request, requested)
        if full["sim"].safe and "full_payment" in self.engine.accepted_methods(profile): candidates.append(full)
        candidates.extend(self._option_candidates(request, profile))
        partial = self._partial_candidate(request, today_safe, earliest)
        if partial and "partial_payment" in self.engine.accepted_methods(profile): candidates.append(partial)
        if not candidates and "full_payment" in self.engine.accepted_methods(profile):
            candidates.extend(self._change_candidates(request, full))

        candidate_summary = [{"method": c["kind"], "total": c["total"], "safe": c["sim"].safe,
                              "minimum_balance": c["sim"].minimum_balance, "changes": len(c.get("changes", []))} for c in candidates]
        self.reasoner.generate_strategy_rationale(request.to_dict(), candidate_summary)

        if candidates:
            best = min(candidates, key=lambda x: self._rank(x, request))
            method = best["kind"]
            status = "affordable_now" if method == "full_payment" and best["payments"][0].date == pd.Timestamp(request.request_date) else "affordable_with_plan"
            if method == "full_payment" and best.get("option") is None and earliest is not None and earliest > pd.Timestamp(request.request_date):
                status, method = "affordable_later", "wait"
            plan = "|".join(f"{p.date:%Y-%m-%d}:{self._money(p.amount):.2f}" for p in best["payments"])
            changes = best.get("changes", [])
            change_text = "none" if not changes else "|".join(
                f"stop:{c['event_id']}" if c["action"] == "stop" else f"reduce_to:{c['event_id']}:{self._money(c['new_amount']):.2f}" for c in changes)
            explanation = f"Verified {method.replace('_', ' ')}. Projected minimum balance is {self._money(best['sim'].minimum_balance):.2f} versus required {self._money(profile.minimum_balance_to_keep):.2f}."
            if changes: explanation += " Flexible recurring spending changes preserve the required safety buffer."
            result = {"request_id": request.request_id, "amount_safe_to_pay": self._money(today_safe),
                      "affordability_status": status, "recommended_payment_method": method,
                      "payment_plan": plan, "earliest_date_for_full_payment": "" if earliest is None else f"{earliest:%Y-%m-%d}",
                      "spending_changes_needed": change_text, "decision_explanation": explanation}
        elif earliest is not None and "full_payment" in self.engine.accepted_methods(profile):
            result = {"request_id": request.request_id, "amount_safe_to_pay": self._money(today_safe),
                      "affordability_status": "affordable_later", "recommended_payment_method": "wait",
                      "payment_plan": f"{earliest:%Y-%m-%d}:{requested:.2f}", "earliest_date_for_full_payment": f"{earliest:%Y-%m-%d}",
                      "spending_changes_needed": "none", "decision_explanation": f"Wait until {earliest:%Y-%m-%d}; paying earlier would breach the minimum balance constraint."}
        else:
            result = {"request_id": request.request_id, "amount_safe_to_pay": self._money(today_safe),
                      "affordability_status": "not_affordable", "recommended_payment_method": "not_recommended",
                      "payment_plan": "none", "earliest_date_for_full_payment": "" if earliest is None else f"{earliest:%Y-%m-%d}",
                      "spending_changes_needed": "none", "decision_explanation": "Do not proceed. No eligible plan completes the request safely while maintaining the minimum balance."}

        self.reasoner.explain(result, {"risk": risk, "candidate_count": len(candidates), "verified": True})
        self.audit_traces[str(request.request_id)] = {"risk": risk, "candidates": candidate_summary,
                                                      "llm_trace": self.reasoner.trace.stages[-4:]}
        return result
