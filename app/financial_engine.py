from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import math
import re
from typing import Any

import pandas as pd

from app.image_layer import ImageEvidence

IGNORED = {"cancelled", "failed", "unrealized", "canceled"}


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
    """Deterministic 90-day financial engine with indexed/cached state reconstruction."""

    def __init__(self, data: dict[str, pd.DataFrame], horizon_days: int = 90, dataset_dir: str = "dataset"):
        self.data = data
        self.horizon_days = horizon_days
        self.events = data["events"].copy()
        self.events["event_date"] = pd.to_datetime(self.events["event_date"], errors="coerce")
        self.events["settlement_date"] = pd.to_datetime(self.events["settlement_date"], errors="coerce")
        self.events = self.events[self.events.event_date.notna()].copy()
        self.profiles = data["profiles"]
        self.requests = data["requests"]
        self.options = data["payment_options"]
        self.messages = data["messages"]
        self.images = data["images"]
        self.exchange_rates = data["exchange_rates"]
        self.image_evidence = ImageEvidence(dataset_dir)

        self._profiles_by_user = {str(row.user_id): row for _, row in self.profiles.iterrows()}
        self._options_by_request = {str(k): g for k, g in self.options.groupby("request_id", sort=False)}
        self._event_amount_cache: dict[str, float] = {}
        self._forecast_cache: dict[tuple[str, str], tuple[pd.DatetimeIndex, list[float]]] = {}
        self._simulation_cache: dict[tuple[str, str, tuple[tuple[str, float], ...]], Simulation] = {}
        self._fx_cache: dict[tuple[str, str, str], float] = {}
        self._message_overrides: dict[str, dict[str, Any]] = self._build_message_overrides()
        self._events_by_user: dict[str, pd.DataFrame] = {}
        self._rebuild_event_index()

    @staticmethod
    def _cats(value: Any) -> set[str]:
        if pd.isna(value):
            return set()
        return {x.strip().lower() for x in str(value).split("|") if x.strip()}

    @staticmethod
    def _first_existing(columns: Any, names: tuple[str, ...]) -> str | None:
        lookup = {str(c).lower(): str(c) for c in columns}
        for name in names:
            if name.lower() in lookup:
                return lookup[name.lower()]
        return None

    def profile(self, user_id: str) -> pd.Series:
        row = self._profiles_by_user.get(str(user_id))
        if row is None:
            raise ValueError(f"No profile for {user_id}")
        return row

    def _build_message_overrides(self) -> dict[str, dict[str, Any]]:
        """Resolve explicit message amendments/cancellations tied to supplied events."""
        result: dict[str, dict[str, Any]] = {}
        if self.messages.empty or "related_event_id" not in self.messages.columns:
            return result
        for _, msg in self.messages.iterrows():
            event_id = msg.get("related_event_id")
            if pd.isna(event_id) or str(event_id).strip() == "":
                continue
            eid = str(event_id)
            text = str(msg.get("message_text", "")).lower()
            item = result.setdefault(eid, {"cancelled": False, "amount": None})
            if re.search(r"\b(cancel|cancelled|canceled|void|reversed|do not pay|don't pay)\b", text):
                item["cancelled"] = True
            if re.search(r"\b(settled|confirmed|approved|completed)\b", text):
                item["force_confirmed"] = True
            # Explicit amount amendments in the message take precedence over an estimate.
            if re.search(r"\b(amend|amended|updated|update|changed|change|correct|correction|new amount)\b", text):
                nums = re.findall(r"(?<![A-Za-z])\d[\d,]*(?:\.\d+)?", text)
                if nums:
                    try:
                        item["amount"] = float(nums[-1].replace(",", ""))
                    except ValueError:
                        pass
        return result

    def _rebuild_event_index(self) -> None:
        """Apply message-level lifecycle overrides and index events by user."""
        rows = []
        for _, row in self.events.iterrows():
            eid = str(row.event_id)
            override = self._message_overrides.get(eid, {})
            status = str(row.status).lower()
            if status in IGNORED or override.get("cancelled"):
                continue
            r = row.copy()
            if override.get("force_confirmed") and str(r.get("status", "")).lower() == "pending":
                r["status"] = "confirmed"
            if override.get("amount") is not None:
                r["amount"] = float(override["amount"])
            rows.append(r)
        self.events = pd.DataFrame(rows, columns=self.events.columns) if rows else self.events.iloc[0:0].copy()
        self._events_by_user = {
            str(user_id): group.sort_values("event_date")
            for user_id, group in self.events.groupby("user_id", sort=False)
        }

    def _user_events(self, user_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        e = self._events_by_user.get(str(user_id))
        if e is None:
            return self.events.iloc[0:0]
        return e[(e.event_date >= start) & (e.event_date <= end)]

    def _fx_rate(self, date: pd.Timestamp, from_currency: str, to_currency: str) -> float:
        src = str(from_currency).upper().strip()
        dst = str(to_currency).upper().strip()
        if not src or not dst or src == dst:
            return 1.0
        key = (f"{pd.Timestamp(date).date()}", src, dst)
        if key in self._fx_cache:
            return self._fx_cache[key]
        df = self.exchange_rates
        if df.empty:
            raise ValueError(f"Missing exchange rate data for {src}->{dst}")
        date_col = self._first_existing(df.columns, ("rate_date", "date", "exchange_rate_date"))
        from_col = self._first_existing(df.columns, ("from_currency", "source_currency", "base_currency"))
        to_col = self._first_existing(df.columns, ("to_currency", "target_currency", "quote_currency"))
        rate_col = self._first_existing(df.columns, ("rate", "exchange_rate", "conversion_rate"))
        if not all((date_col, from_col, to_col, rate_col)):
            raise ValueError("exchange_rates.csv does not contain recognizable rate columns")
        dates = pd.to_datetime(df[date_col], errors="coerce").dt.date
        exact = df[(dates == pd.Timestamp(date).date()) & (df[from_col].astype(str).str.upper() == src) & (df[to_col].astype(str).str.upper() == dst)]
        if not exact.empty:
            rate = float(exact.iloc[0][rate_col])
            self._fx_cache[key] = rate
            return rate
        # Support an inverse supplied rate without inventing a live rate.
        inverse = df[(dates == pd.Timestamp(date).date()) & (df[from_col].astype(str).str.upper() == dst) & (df[to_col].astype(str).str.upper() == src)]
        if not inverse.empty:
            rate = 1.0 / float(inverse.iloc[0][rate_col])
            self._fx_cache[key] = rate
            return rate
        raise ValueError(f"No dated exchange rate for {src}->{dst} on {pd.Timestamp(date).date()}")

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
        if math.isnan(amount):
            self._event_amount_cache[event_id] = amount
            return amount
        currency_col = self._first_existing(row.index, ("currency", "event_currency", "transaction_currency"))
        event_currency = str(row.get(currency_col, "")) if currency_col else ""
        home = str(self.profile(row.user_id).get("home_currency", ""))
        if event_currency and home and event_currency.upper() != home.upper():
            amount *= self._fx_rate(pd.Timestamp(row.event_date), event_currency, home)
        self._event_amount_cache[event_id] = amount
        return amount

    def _forecast(self, user_id: str, request_date: str) -> tuple[pd.DatetimeIndex, list[float]]:
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

    def _simulate_from_forecast(self, user_id: str, request_date: str, payments: list[Payment], changes: list[dict[str, Any]] | None = None) -> Simulation:
        dates, base_balances = self._forecast(user_id, request_date)
        p = self.profile(user_id)
        minimum_required = float(p["minimum_balance_to_keep"])
        changes_by_id = {str(x["event_id"]): x for x in (changes or [])}
        payment_by_day: dict[pd.Timestamp, float] = {}
        for pay in payments:
            day = pd.Timestamp(pay.date).normalize()
            if dates[0] <= day <= dates[-1]:
                payment_by_day[day] = payment_by_day.get(day, 0.0) + float(pay.amount)
        if not changes_by_id:
            minimum = min([float(p["current_available_balance"])] + [float(b) for b in base_balances])
            for day, base in zip(dates, base_balances):
                minimum = min(minimum, float(base) - payment_by_day.get(day, 0.0))
            ending = float(base_balances[-1]) - sum(payment_by_day.values()) if base_balances else float(p["current_available_balance"])
            shortfall = max(0.0, minimum_required - minimum)
            return Simulation(shortfall <= 1e-7, minimum, ending, shortfall)

        events = self._user_events(user_id, dates[0], dates[-1])
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
        return min(float(requested), max(0.0, round(minimum - required, 10)))

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

    def _is_recurring(self, row: pd.Series) -> bool:
        col = self._first_existing(row.index, ("recurring", "is_recurring", "recurrence", "frequency", "event_frequency"))
        if col is None:
            return True
        value = row.get(col)
        if pd.isna(value):
            return True
        text = str(value).lower().strip()
        return text not in {"false", "0", "no", "none", "one_time", "one-time"}

    def flexible_changes(self, user_id: str, request_date: str, requested: float, deadline: str) -> list[dict[str, Any]]:
        p = self.profile(user_id)
        reducible = self._cats(p.expense_categories_user_is_willing_to_reduce)
        stoppable = self._cats(p.expense_categories_user_is_willing_to_stop)
        e = self._user_events(user_id, pd.Timestamp(request_date), pd.Timestamp(request_date) + timedelta(days=self.horizon_days))
        e = e[(e.direction == "debit") & e.flexibility.isin(["reducible", "stoppable"])]
        changes = []
        for _, row in e.iterrows():
            if not self._is_recurring(row):
                continue
            amount = self._event_amount(row)
            if math.isnan(amount) or amount <= 0:
                continue
            category = str(row.category).lower()
            flexibility = str(row.flexibility).lower()
            if category in stoppable and flexibility == "stoppable":
                changes.append({"event_id": row.event_id, "category": row.category, "amount": amount, "new_amount": 0.0, "action": "stop"})
            elif category in reducible and flexibility == "reducible":
                minimum = row.get("minimum_allowed_amount")
                new_amount = float(minimum) if pd.notna(minimum) else amount * 0.5
                changes.append({"event_id": row.event_id, "category": row.category, "amount": amount, "new_amount": max(0.0, min(amount, new_amount)), "action": "reduce"})
        return changes

    def adjusted_simulation(self, user_id: str, request_date: str, payments: list[Payment], changes: list[dict[str, Any]]) -> Simulation:
        return self._simulate_from_forecast(user_id, request_date, payments, changes)
