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
    """Hard-constraint financial engine. LLMs never decide safety.

    Performance design: event rows, payment options and per-user forecasts are
    indexed/cached once. Simulations no longer repeatedly scan the full events
    DataFrame or filter it once per day.
    """

    def __init__(self, data: dict[str, pd.DataFrame], horizon_days: int = 90, dataset_dir: str = "dataset"):
        self.data = data
        self.horizon_days = horizon_days
        self.events = data["events"].copy()
        self.events["event_date"] = pd.to_datetime(self.events["event_date"], errors="coerce")
        self.events["settlement_date"] = pd.to_datetime(self.events["settlement_date"], errors="coerce")
        self.events = self.events[
            self.events.event_date.notna()
            & ~self.events.status.astype(str).str.lower().isin(IGNORED)
        ].copy()
        self.profiles = data["profiles"]
        self.requests = data["requests"]
        self.options = data["payment_options"]
        self.messages = data["messages"]
        self.images = data["images"]
        self.image_evidence = ImageEvidence(dataset_dir)

        self._profiles_by_user = {
            str(row.user_id): row for _, row in self.profiles.iterrows()
        }
        self._options_by_request = {
            str(request_id): group
            for request_id, group in self.options.groupby("request_id", sort=False)
        }
        self._events_by_user: dict[str, pd.DataFrame] = {
            str(user_id): group.sort_values("event_date")
            for user_id, group in self.events.groupby("user_id", sort=False)
        }
        self._event_amount_cache: dict[str, float] = {}
        self._forecast_cache: dict[tuple[str, str], tuple[pd.DatetimeIndex, list[float]]] = {}
        self._simulation_cache: dict[tuple[str, str, tuple[tuple[str, float], ...]], Simulation] = {}

    @staticmethod
    def _cats(value: Any) -> set[str]:
        if pd.isna(value):
            return set()
        return {x.strip() for x in str(value).split("|") if x.strip()}

    def profile(self, user_id: str) -> pd.Series:
        row = self._profiles_by_user.get(str(user_id))
        if row is None:
            raise ValueError(f"No profile for {user_id}")
        return row

    def _message_adjustments(self, user_id: str, request_id: str) -> dict[str, Any]:
        rows = self.messages[self.messages.user_id.eq(user_id) & (self.messages.request_id.eq(request_id) | self.messages.request_id.isna())]
        text = " ".join(rows.message_text.fillna("").astype(str).tolist()).lower()
        salary = None
        m = re.search(r"(?:salary|pay|gaji|payroll)[^0-9]{0,40}([0-9][0-9,]*(?:\.\d+)?)", text)
        if m:
            try:
                salary = float(m.group(1).replace(",", ""))
            except ValueError:
                pass
        return {"text": text, "salary_hint": salary}

    def _user_events(self, user_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        e = self._events_by_user.get(str(user_id))
        if e is None:
            return self.events.iloc[0:0]
        return e[(e.event_date >= start) & (e.event_date <= end)]

    def _event_amount(self, row: pd.Series) -> float:
        event_id = str(row.event_id)
        cached = self._event_amount_cache.get(event_id)
        if cached is not None:
            return cached
        value = row.get("amount")
        if pd.notna(value) and str(value).strip() != "":
            amount = float(value)
        else:
            extracted = self.image_evidence.amount_for_event(event_id)
            amount = math.nan if extracted is None else float(extracted)
        self._event_amount_cache[event_id] = amount
        return amount

    def _forecast(self, user_id: str, request_date: str) -> tuple[pd.DatetimeIndex, list[float]]:
        """Cached 90-day end-of-day balances with no new request payment."""
        key = (str(user_id), str(pd.Timestamp(request_date).date()))
        cached = self._forecast_cache.get(key)
        if cached is not None:
            return cached

        start = pd.Timestamp(request_date).normalize()
        end = start + timedelta(days=self.horizon_days)
        dates = pd.date_range(start, end, freq="D")
        balance = float(self.profile(user_id)["current_available_balance"])
        events = self._user_events(user_id, start, end)
        by_day: dict[pd.Timestamp, list[pd.Series]] = {}
        for _, row in events.iterrows():
            by_day.setdefault(row.event_date.normalize(), []).append(row)

        balances: list[float] = []
        for day in dates:
            for row in by_day.get(day, ()):
                amount = self._event_amount(row)
                if math.isnan(amount):
                    if str(row.direction).lower() == "debit":
                        balance = -math.inf
                        break
                    continue
                direction = str(row.direction).lower()
                status = str(row.status).lower()
                if direction == "credit" and status != "pending":
                    balance += amount
                elif direction == "debit":
                    balance -= amount
            balances.append(balance)
            if balance == -math.inf:
                balances.extend([-math.inf] * (len(dates) - len(balances)))
                break

        result = (dates, balances)
        self._forecast_cache[key] = result
        return result

    def _simulate_from_forecast(
        self,
        user_id: str,
        request_date: str,
        payments: list[Payment],
        changes: list[dict[str, Any]] | None = None,
    ) -> Simulation:
        start = pd.Timestamp(request_date).normalize()
        dates, base_balances = self._forecast(user_id, request_date)
        p = self.profile(user_id)
        minimum_required = float(p["minimum_balance_to_keep"])
        changes_by_id = {str(x["event_id"]): x for x in (changes or [])}

        if not changes_by_id:
            payment_by_day: dict[pd.Timestamp, float] = {}
            for pay in payments:
                day = pd.Timestamp(pay.date).normalize()
                if dates[0] <= day <= dates[-1]:
                    payment_by_day[day] = payment_by_day.get(day, 0.0) + float(pay.amount)
            minimum = float(p["current_available_balance"])
            for day, base in zip(dates, base_balances):
                value = base - payment_by_day.get(day, 0.0)
                if value < minimum:
                    minimum = value
            total_payment = sum(payment_by_day.values())
            ending = float(base_balances[-1]) - total_payment if base_balances else float(p["current_available_balance"])
            shortfall = max(0.0, minimum_required - minimum)
            return Simulation(shortfall <= 1e-7, minimum, ending, shortfall)

        events = self._user_events(user_id, dates[0], dates[-1])
        payment_by_day: dict[pd.Timestamp, float] = {}
        for pay in payments:
            day = pd.Timestamp(pay.date).normalize()
            if dates[0] <= day <= dates[-1]:
                payment_by_day[day] = payment_by_day.get(day, 0.0) + float(pay.amount)

        balance = float(p["current_available_balance"])
        minimum = balance
        by_day: dict[pd.Timestamp, list[pd.Series]] = {}
        for _, row in events.iterrows():
            by_day.setdefault(row.event_date.normalize(), []).append(row)
        for day in dates:
            for row in by_day.get(day, ()):
                amount = self._event_amount(row)
                if math.isnan(amount):
                    if str(row.direction).lower() == "debit":
                        return Simulation(False, -math.inf, -math.inf, math.inf)
                    continue
                change = changes_by_id.get(str(row.event_id))
                if change is not None:
                    amount = float(change["new_amount"])
                direction = str(row.direction).lower()
                status = str(row.status).lower()
                if direction == "credit" and status != "pending":
                    balance += amount
                elif direction == "debit":
                    balance -= amount
            balance -= payment_by_day.get(day, 0.0)
            minimum = min(minimum, balance)
        shortfall = max(0.0, minimum_required - minimum)
        return Simulation(shortfall <= 1e-7, minimum, balance, shortfall)

    def simulate(self, user_id: str, request_date: str, payments: list[Payment]) -> Simulation:
        payment_key = tuple(sorted((pd.Timestamp(p.date).isoformat(), round(float(p.amount), 8)) for p in payments))
        key = (str(user_id), str(pd.Timestamp(request_date).date()), payment_key)
        cached = self._simulation_cache.get(key)
        if cached is not None:
            return cached
        result = self._simulate_from_forecast(user_id, request_date, payments)
        self._simulation_cache[key] = result
        return result

    def option_payments(self, option: pd.Series) -> list[Payment]:
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

    def accepted_methods(self, profile: pd.Series) -> set[str]:
        return self._cats(profile.payment_methods_user_will_consider)

    def max_safe_today(self, user_id: str, request_date: str, requested: float) -> float:
        _, balances = self._forecast(user_id, request_date)
        if not balances:
            return 0.0
        current = float(self.profile(user_id)["current_available_balance"])
        minimum = min(current, min(balances))
        required = float(self.profile(user_id)["minimum_balance_to_keep"])
        safe = max(0.0, minimum - required)
        return min(float(requested), round(safe, 10))

    def earliest_full_payment(self, user_id: str, request_date: str, requested: float, deadline: str) -> pd.Timestamp | None:
        start = pd.Timestamp(request_date).normalize()
        end = min(pd.Timestamp(deadline).normalize(), start + timedelta(days=self.horizon_days))
        dates, balances = self._forecast(user_id, request_date)
        required = float(self.profile(user_id)["minimum_balance_to_keep"])
        if float(self.profile(user_id)["current_available_balance"]) < required - 1e-7:
            return None
        for day, balance in zip(dates, balances):
            if start <= day <= end and balance - float(requested) >= required - 1e-7:
                return day
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
            if math.isnan(amount) or amount <= 0:
                continue
            if row.category in stoppable and str(row.flexibility) == "stoppable":
                changes.append({"event_id": row.event_id, "category": row.category, "amount": amount, "new_amount": 0.0, "action": "stop"})
            elif row.category in reducible or str(row.flexibility) == "reducible":
                minimum = row.get("minimum_allowed_amount")
                new_amount = float(minimum) if pd.notna(minimum) else amount * 0.5
                changes.append({"event_id": row.event_id, "category": row.category, "amount": amount, "new_amount": max(0.0, min(amount, new_amount)), "action": "reduce"})
        return changes

    def adjusted_simulation(self, user_id: str, request_date: str, payments: list[Payment], changes: list[dict[str, Any]]) -> Simulation:
        return self._simulate_from_forecast(user_id, request_date, payments, changes)
