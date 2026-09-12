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
    started = time.perf_counter()
    agent = FinancialAgent(ROOT / "dataset")
    requests = agent.data["requests"]
    rows: list[dict] = []

    for i, (_, request) in enumerate(requests.iterrows(), 1):
        print(f"[{i}/{len(requests)}] {request.request_id}")
        rows.append(agent.decide(request))

    verifier = OutputVerifier(agent.data)
    clean, issues = verifier.verify(requests, rows)
    pd.DataFrame(clean, columns=OUTPUT_COLUMNS).to_csv(ROOT / "output.csv", index=False)

    usage_dir = ROOT / "evaluation"
    usage_dir.mkdir(exist_ok=True)
    # Competition mode performs zero LLM calls. Keep usage accounting for an
    # optional Ollama demo without requiring an AI object on FinancialAgent.
    llm_usage = agent.reasoner.ai.usage
    total_in = sum(x.get("input_tokens", 0) for x in llm_usage)
    total_out = sum(x.get("output_tokens", 0) for x in llm_usage)
    elapsed = time.perf_counter() - started

    report = f"""# Agent Usage Report

## Architecture
The competition path is a deterministic hybrid financial agent. Structured request data, indexed financial events, cached 90-day forecasts, payment-plan search, and the independent verifier determine the answer. LLM calls are not required for the final submission path.

## Final dataset run
- Requests: {len(requests)}
- Runtime: {elapsed:.2f} seconds
- LLM calls: {len(llm_usage)}
- Input tokens: {total_in}
- Output tokens: {total_out}
- Total tokens: {total_in + total_out}
- Provider: {llm_usage[0].get('provider', 'none') if llm_usage else 'none'}
- Model: {llm_usage[0].get('model', 'none') if llm_usage else 'none'}
- Estimated API cost: $0 in deterministic competition mode
- Verification issues: {len(issues)}

## Agent stages
1. Structured Request Intake
2. Indexed Financial State + 90-Day Cash-Flow Forecast
3. Payment Strategy / Plan Search
4. Optional Risk Model
5. Independent Output Verifier
6. Deterministic Decision Explanation

## Runtime optimizations
- Financial events are indexed by user once.
- Payment options are indexed by request once.
- Event amounts, including image-derived amounts, are cached.
- 90-day balance forecasts are cached per user/request date.
- Repeated payment simulations are cached.
- The previous 48-iteration safe-amount binary search is replaced by a direct forecast calculation.
- The previous per-request LLM explanation call is removed from the competition path.
- Flexible-spending search is bounded to avoid combinatorial explosion.

## Safety boundary
LLM output cannot approve a purchase, calculate affordability, modify balances, ignore minimum-balance rules, or override verification. Messages and OCR output remain untrusted evidence.
"""
    (usage_dir / "usage_report.md").write_text(report, encoding="utf-8")

    if issues:
        print(json.dumps({"verification_issues": issues}, indent=2))
    print(f"\nWrote {ROOT / 'output.csv'}")
    print(f"Elapsed: {elapsed:.2f}s | LLM calls: {len(llm_usage)} | verifier issues: {len(issues)}")


if __name__ == "__main__":
    main()
