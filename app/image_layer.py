from __future__ import annotations

import re
from pathlib import Path


class ImageEvidence:
    """Optional OCR for the small set of challenge images containing missing amounts."""

    def __init__(self, dataset_dir: str | Path):
        self.dataset_dir = Path(dataset_dir)
        self.image_map = {}
        csv_path = self.dataset_dir / "images.csv"
        if csv_path.exists():
            import pandas as pd
            df = pd.read_csv(csv_path)
            self.image_map = {str(r.related_event_id): str(r.image_id) for _, r in df.iterrows()}

    def amount_for_event(self, event_id: str) -> float | None:
        image_id = self.image_map.get(str(event_id))
        if not image_id:
            return None
        candidates = [
            self.dataset_dir / "media" / "images" / f"{image_id}.png",
            self.dataset_dir / "media" / f"{image_id}.png",
            self.dataset_dir / f"{image_id}.png",
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            return None
        try:
            from PIL import Image
            import pytesseract
            text = pytesseract.image_to_string(Image.open(path), config="--psm 6")
        except Exception:
            return None
        nums = re.findall(r"(?<![A-Za-z])\d[\d,]*(?:\.\d+)?", text)
        values = []
        for n in nums:
            try: values.append(float(n.replace(",", "")))
            except ValueError: pass
        return max(values) if values else None
