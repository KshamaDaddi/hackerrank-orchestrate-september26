from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import math
import re
from typing import Any

import pandas as pd

from app.image_layer import ImageEvidence

IGNORED = {"cancelled", "failed", "unrealized"}


@dataclass(frozen=True)
class Payment:
    date: pd.Timestamp
    amount: float
    label: str = "request"


@dataclass
class Simulation:
    safe: bool
    minimum_balance: float
    ending_balance: float
    shortfall: float


class FinancialEngine:
    """Hard-constraint financial engine. LLMs never decide safety."""

    def __init__(self, data: dict[str, pd.DataFrame], horizon_days: int = 90, dataset_dir: str = "dataset"):
        self.data = data
        self.horizon_days = horizon_days
        self.events = data["events"].copy()
        self.events["event_date"] = pd.to_datetime(self.events["event_date"], errors="coerce")
        self.events["settlement_date"] = pd.to_datetime(self.events["settlement_date"], errors="coerce")
        self.profiles = data["profiles"]
        self.requests = data["requests"]
        self.options = data["payment_options"]
        self.messages = data["messages"]
        self.images = data["images"]
        self.image_evidence = ImageEvidence(dataset_dir)

    @staticmethod
    def _cats(value: Any) -> set[str]:
        if pd.isna(value): return set()
        return {x.strip() for x in str(value).split("|") if x.strip()}

    def profile(self, user_id: str) -> pd.Series:
        rows = self.profiles[self.profiles.user_id.eq(user_id)]
        if rows.empty: raise ValueError(f"No profile for {user_id}")
        return rows.iloc[0]

    def _message_adjustments(self, user_id: str, request_id: str) -> dict[str, Any]:
        rows = self.messages[self.messages.user_id.eq(user_id) & (self.messages.request_id.eq(request_id) | self.messages.request_id.isna())]
        text = " ".join(rows.message_text.fillna("").astype(str).tolist()).lower()
        salary = None
        m = re.search(r"(?:salary|pay|gaji|payroll)[^0-9]{0,40}([0-9][0-9,]*(?:\.\d+)?)", text)
        if m:
            try: salary = float(m.group(1).replace(",", ""))
            except ValueError: pass
        return {"text": text, "salary_hint": salary}

    def _user_events(self, user_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        e = self.events[self.events.user_id.eq(user_id)].copy()
        e = e[~e.status.astype(str).str.lower().isin(IGNORED)]
        e = e[e.event_date.notna()]
        return e[(e.event_date >= start) & (e.event_date <= end)]

    def _event_amount(self, row: pd.Series) -> float:
        value = row.get("amount")
        if pd.notna(value) and str(value).strip() != "": return float(value)
        # The challenge explicitly says a blank amount is not zero. Try the linked image.
        extracted = self.image_evidence.amount_for_event(str(row.event_id))
        return math.nan if extracted is None else float(extracted)

    def simulate(self, user_id: str, request_date: str, payments: list[Payment]) -> Simulation:
        start = pd.Timestamp(request_date); end = start + timedelta(days=self.horizon_days)
        p = self.profile(user_id); minimum_required = float(p["minimum_balance_to_keep"]); balance = float(p["current_available_balance"])
        events = self._user_events(user_id, start, end)
        payment_map: dict[pd.Timestamp, float] = {}
        for pay in payments:
            if start <= pay.date <= end: payment_map[pay.date.normalize()] = payment_map.get(pay.date.normalize(), 0.0) + pay.amount
        minimum = balance
        for day in pd.date_range(start.normalize(), end.normalize(), freq="D"):
            for _, row in events[events.event_date.dt.normalize().eq(day)].iterrows():
                amount = self._event_amount(row)
                if math.isnan(amount):
                    # Unknown debit is treated conservatively as a safety failure.
                    if str(row.direction).lower() == "debit": return Simulation(False, -math.inf, -math.inf, math.inf)
                    continue
                direction = str(row.direction).lower(); status = str(row.status).lower()
                # Confirmed/scheduled income is usable. Pending credits are explicitly excluded.
                if direction == "credit" and status != "pending": balance += amount
                elif direction == "debit": balance -= amount
            balance -= payment_map.get(day, 0.0); minimum = min(minimum, balance)
        shortfall = max(0.0, minimum_required - minimum)
        return Simulation(shortfall <= 1e-7, minimum, balance, shortfall)

    def option_payments(self, option: pd.Series) -> list[Payment]:
        """Convert a supplied payment option into payments.

        The official dataset contains a small number of malformed/orphan option
        rows with missing schedule fields. Such rows are not usable payment
        plans and must be ignored rather than crashing the complete evaluation.
        """
        required = ("first_payment_date", "number_of_payments", "payment_amount", "payment_frequency_days")
        if any(col not in option.index or pd.isna(option[col]) for col in required):
            return []
        try:
            first = pd.Timestamp(option.first_payment_date)
            n = int(option.number_of_payments)
            amount = float(option.payment_amount)
            freq = int(option.payment_frequency_days)
        except (TypeError, ValueError, OverflowError):
            return []
        if pd.isna(first) or n <= 0 or not math.isfinite(amount) or amount <= 0 or freq <= 0:
            return []
        return [Payment(first + timedelta(days=i * freq), amount, str(option.payment_option_id)) for i in range(n)]

    def accepted_methods(self, profile: pd.Series) -> set[str]: return self._cats(profile.payment_methods_user_will_consider)

    def max_safe_today(self, user_id: str, request_date: str, requested: float) -> float:
        lo, hi = 0.0, max(0.0, requested)
        for _ in range(48):
            mid = (lo + hi) / 2.0
            if self.simulate(user_id, request_date, [Payment(pd.Timestamp(request_date), mid)]).safe: lo = mid
            else: hi = mid
        return min(requested, round(lo, 10))

    def earliest_full_payment(self, user_id: str, request_date: str, requested: float, deadline: str) -> pd.Timestamp | None:
        start = pd.Timestamp(request_date); end = min(pd.Timestamp(deadline), start + timedelta(days=self.horizon_days))
        for day in pd.date_range(start, end, freq="D"):
            if self.simulate(user_id, request_date, [Payment(day, requested, "wait")]).safe: return day
        return None

    def flexible_changes(self, user_id: str, request_date: str, requested: float, deadline: str) -> list[dict[str, Any]]:
        p = self.profile(user_id)
        reducible = self._cats(p.expense_categories_user_is_willing_to_reduce)
        stoppable = self._cats(p.expense_categories_user_is_willing_to_stop)
        e = self._user_events(user_id, pd.Timestamp(request_date), pd.Timestamp(request_date) + timedelta(days=self.horizon_days))
        e = e[(e.direction == "debit") & e.flexibility.isin(["reducible", "stoppable"])]
        changes = []
        for _, row in e.iterrows():
            amount = self._event_amount(row)
            if math.isnan(amount) or amount <= 0: continue
            if row.category in stoppable and str(row.flexibility) == "stoppable":
                changes.append({"event_id": row.event_id, "category": row.category, "amount": amount, "new_amount": 0.0, "action": "stop"})
            elif row.category in reducible or str(row.flexibility) == "reducible":
                minimum = row.get("minimum_allowed_amount")
                new_amount = float(minimum) if pd.notna(minimum) else amount * 0.5
                changes.append({"event_id": row.event_id, "category": row.category, "amount": amount, "new_amount": max(0.0, min(amount, new_amount)), "action": "reduce"})
        return changes

    def adjusted_simulation(self, user_id: str, request_date: str, payments: list[Payment], changes: list[dict[str, Any]]) -> Simulation:
        start = pd.Timestamp(request_date); end = start + timedelta(days=self.horizon_days)
        p = self.profile(user_id); minimum_required = float(p.minimum_balance_to_keep); balance = float(p.current_available_balance)
        events = self._user_events(user_id, start, end); changes_by_id = {str(x["event_id"]): x for x in changes}
        payment_map: dict[pd.Timestamp, float] = {}
        for pay in payments: payment_map[pay.date.normalize()] = payment_map.get(pay.date.normalize(), 0.0) + pay.amount
        minimum = balance
        for day in pd.date_range(start.normalize(), end.normalize(), freq="D"):
            for _, row in events[events.event_date.dt.normalize().eq(day)].iterrows():
                amount = self._event_amount(row)
                if math.isnan(amount):
                    if str(row.direction).lower() == "debit": return Simulation(False, -math.inf, -math.inf, math.inf)
                    continue
                change = changes_by_id.get(str(row.event_id))
                if change: amount = float(change["new_amount"])
                if str(row.direction).lower() == "credit" and str(row.status).lower() != "pending": balance += amount
                elif str(row.direction).lower() == "debit": balance -= amount
            balance -= payment_map.get(day, 0.0); minimum = min(minimum, balance)
        shortfall = max(0.0, minimum_required - minimum)
        return Simulation(shortfall <= 1e-7, minimum, balance, shortfall)
