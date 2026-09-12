from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class RiskModel:
    """Advisory anomaly/risk layer. Hard safety constraints remain deterministic."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = IsolationForest(n_estimators=150, contamination="auto", random_state=42)
        self.fitted = False

    def fit(self, profiles: pd.DataFrame) -> None:
        cols = ["current_available_balance", "minimum_balance_to_keep"]
        x = profiles[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        x["buffer_ratio"] = x[cols[0]] / np.maximum(x[cols[1]], 1e-9)
        self.model.fit(self.scaler.fit_transform(x))
        self.fitted = True

    def score(self, profile: pd.Series) -> float:
        if not self.fitted:
            return 0.5
        x = pd.DataFrame([{
            "current_available_balance": float(profile.current_available_balance),
            "minimum_balance_to_keep": float(profile.minimum_balance_to_keep),
            "buffer_ratio": float(profile.current_available_balance) / max(float(profile.minimum_balance_to_keep), 1e-9),
        }])
        raw = float(self.model.decision_function(self.scaler.transform(x))[0])
        return round(float(np.clip(0.5 - raw, 0.0, 1.0)), 4)
