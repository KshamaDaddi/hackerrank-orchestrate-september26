from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent import FinancialAgent
from app.verifier import OutputVerifier

OUTPUT_COLUMNS = [
    "request_id", "amount_safe_to_pay", "affordability_status",
    "recommended_payment_method", "payment_plan",
    "earliest_date_for_full_payment", "spending_changes_needed",
    "decision_explanation",
]


def main() -> None:
    start = time.perf_counter()
    agent = FinancialAgent(ROOT / "dataset")
    requests = agent.data["requests"]
    rows = []
    for i, (_, request) in enumerate(requests.iterrows(), 1):
        print(f"[{i}/{len(requests)}] {request.request_id}")
        rows.append(agent.decide(request))

    verifier = OutputVerifier(agent.data)
    clean, issues = verifier.verify(requests, rows)
    out = pd.DataFrame(clean, columns=OUTPUT_COLUMNS)
    out.to_csv(ROOT / "output.csv", index=False)

    usage_dir = ROOT / "evaluation"
    usage_dir.mkdir(exist_ok=True)
    elapsed = time.perf_counter() - start
    (usage_dir / "usage_report.md").write_text(
        "# Model / Usage Report\n\n"
        "## Final run\n"
        f"- Requests: {len(requests)}\n"
        f"- Runtime: {elapsed:.2f} seconds\n"
        "- Deterministic financial calculations: 0 LLM calls\n"
        f"- Optional local Ollama calls: {'enabled if Ollama is reachable' if agent.ai.available() else '0 (Ollama unavailable; heuristic parser used)'}\n"
        f"- Verification issues: {len(issues)}\n"
        "- API cost: $0 for the deterministic pipeline\n\n"
        "The optional Ollama interpreter is used only for semantic hints. It cannot choose a financial plan or override deterministic safety checks. If Ollama is used for a final submission run, record the exact model and token counts from the local Ollama logs before submitting.\n\n"
        "## Security\n"
        "Messages and image-derived text are treated as untrusted evidence. The agent does not execute instructions found inside them.\n"
    , encoding="utf-8")
    if issues:
        print(json.dumps({"verification_issues": issues}, indent=2))
    print(f"\nWrote {ROOT / 'output.csv'}")
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
