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
    llm_usage = agent.ai.usage + agent.reasoner.ai.usage
    total_in = sum(x.get("input_tokens", 0) for x in llm_usage)
    total_out = sum(x.get("output_tokens", 0) for x in llm_usage)
    elapsed = time.perf_counter() - started

    report = f"""# Agent Usage Report

## Architecture
The submission uses a hybrid multi-agent architecture. Specialized LLM agents handle request understanding, evidence interpretation, strategy rationale, and explanation. They are advisory only. The deterministic financial engine performs every monetary calculation and the verifier is authoritative.

## Final dataset run
- Requests: {len(requests)}
- Runtime: {elapsed:.2f} seconds
- LLM calls: {len(llm_usage)}
- Input tokens: {total_in}
- Output tokens: {total_out}
- Total tokens: {total_in + total_out}
- Provider: {llm_usage[0].get('provider', 'none') if llm_usage else 'none'}
- Model: {llm_usage[0].get('model', 'none') if llm_usage else 'none'}
- Estimated API cost: $0 when using the local deterministic/Ollama path
- Verification issues: {len(issues)}

## Agent stages
1. Request Understanding Agent
2. Evidence Agent
3. Deterministic Financial State + 90-Day Cash-Flow Engine
4. Payment Strategy / Plan Search
5. Risk Model
6. Independent Output Verifier
7. Explanation Agent

## Safety boundary
The LLM cannot approve a purchase, calculate affordability, modify balances, ignore minimum-balance rules, or override verification. Messages and OCR output are untrusted evidence and are never executed as instructions.
"""
    (usage_dir / "usage_report.md").write_text(report, encoding="utf-8")

    if issues:
        print(json.dumps({"verification_issues": issues}, indent=2))
    print(f"\nWrote {ROOT / 'output.csv'}")
    print(f"Elapsed: {elapsed:.2f}s | LLM calls: {len(llm_usage)} | verifier issues: {len(issues)}")


if __name__ == "__main__":
    main()
