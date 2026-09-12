from pathlib import Path
import pandas as pd


class DataLoader:
    """Loads and lightly normalizes the participant-facing challenge files."""

    REQUIRED = {
        "requests": "requests.csv",
        "profiles": "financial_profiles.csv",
        "events": "financial_events.csv",
        "messages": "messages.csv",
        "images": "images.csv",
        "payment_options": "request_payment_options.csv",
        "exchange_rates": "exchange_rates.csv",
    }

    def __init__(self, dataset_dir: str | Path = "dataset"):
        self.dataset_dir = Path(dataset_dir)

    def load_all(self) -> dict[str, pd.DataFrame]:
        missing = [f for f in self.REQUIRED.values() if not (self.dataset_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                "Missing dataset files: " + ", ".join(missing)
            )
        return {
            key: pd.read_csv(self.dataset_dir / filename)
            for key, filename in self.REQUIRED.items()
        }

    def load_samples(self) -> pd.DataFrame:
        path = self.dataset_dir / "sample_requests.csv"
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
