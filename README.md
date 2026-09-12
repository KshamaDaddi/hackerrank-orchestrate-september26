# FinAgent-AI — HackerRank Orchestrate September 2026

Hybrid AI financial affordability agent for the **Buy or Wait?** challenge.

## Architecture

```text
Request
  -> optional local Ollama semantic parser
  -> financial state + events + messages + image evidence
  -> deterministic 90-day cash-flow simulator
  -> payment-option / partial-payment search
  -> optional ML anomaly-risk score
  -> deterministic output verifier
  -> output.csv
```

**Core safety rule:** the AI/ML layer is advisory. Money arithmetic, schedules, minimum-balance checks, and final eligibility are deterministic.

## Local setup

Use Python 3.11+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place the official challenge `dataset/` directory in the project root. The runner expects the supplied CSV files and `media/images/*.png` when available.

Run:

```powershell
python code/main.py
```

The required submission file is written to `output.csv`.

## Optional AI mode

The default competition run is deterministic and makes **zero LLM calls**, which keeps the final usage report reproducible.

For a local Ollama demo:

```powershell
$env:USE_OLLAMA="1"
$env:OLLAMA_MODEL="qwen2.5:7b-instruct"
python code/main.py
```

The LLM is constrained to semantic interpretation. It cannot override the financial engine.

## Required output

The runner produces exactly these columns, in order:

`request_id, amount_safe_to_pay, affordability_status, recommended_payment_method, payment_plan, earliest_date_for_full_payment, spending_changes_needed, decision_explanation`

## Robustness

- prompt-injection-resistant separation between evidence and decisions
- hard minimum-balance constraint
- 90-day forecast
- payment-option schedule validation
- partial-payment validation
- flexible-spending guardrails
- deterministic final output validation
- optional local-only LLM; no API key is required
- no secrets committed to the repository

## Submission checklist

1. Run `python code/main.py` on the final dataset.
2. Inspect `output.csv` for the required columns and request count.
3. Inspect `evaluation/usage_report.md`.
4. Create the final `code.zip` containing runnable code, configuration, README, and `evaluation/`.
5. Do not include `.venv`, credentials, or API keys.
