# FinAgent-AI — HackerRank Orchestrate September 2026

> Safety-first financial affordability agent for the **Buy or Wait?** challenge.

FinAgent-AI evaluates whether a requested purchase can be made safely, whether an available payment plan is feasible, or whether the user should wait.

The core design principle is:

**LLMs can interpret and explain. Deterministic code owns the money.**

---

## Problem

The challenge requires reasoning over a user's financial state while respecting:

- current available balance
- future income and expenses
- minimum balance to preserve
- requested purchase amount
- desired completion date
- accepted payment methods
- installment and partial-payment rules
- flexible spending constraints
- evidence from messages and images

A purchase that is affordable today can become unsafe when future expenses arrive. The system therefore evaluates the request against a forward cash-flow projection rather than checking only the current balance.

---

## Solution

For every request, the system:

1. Loads and normalizes the challenge data.
2. Builds the user's financial state.
3. Projects cash flow over a 90-day horizon.
4. Calculates the maximum amount safely payable today.
5. Generates feasible payment candidates.
6. Simulates candidates against hard financial constraints.
7. Selects a deterministic feasible strategy.
8. Runs an independent output verifier.
9. Writes the required output.csv.

Supported strategies include full payment, supplied installments, partial payment, waiting, and eligible flexible-spending changes.

---

## Architecture

    Challenge Dataset
            |
            v
       Data Loader
            |
      +-----+--------------------+
      |                          |
      v                          v
 Financial State          Evidence / AI Layer
      |                          |
      v                          |
  90-Day Forecast               |
      |                          |
      +-----------+--------------+
                  |
                  v
       Candidate Generation
                  |
                  v
       Deterministic Simulation
                  |
                  v
        Candidate Selection
                  |
                  v
       Independent Verifier
                  |
                  v
             output.csv

### Safety boundary

    AI / ML
      |
      +-- interpret evidence
      +-- extract semantics
      +-- provide optional rationale
              |
              v
    Deterministic Financial Engine
      |
      +-- balances
      +-- dates
      +-- payment schedules
      +-- hard constraints
      +-- candidate feasibility
              |
              v
    Independent Verifier
              |
              v
          Final Output

The LLM cannot directly approve an unsafe payment, modify balances, or bypass verification.

---

## Decision pipeline

### 1. Financial state

The agent combines profile information with relevant financial events, including current balance, minimum balance, home currency, accepted payment methods, and eligible flexible-spending categories.

### 2. Event normalization

Events are indexed by user and date. The engine handles credits, debits, transaction status, dated FX conversion, image-derived amounts, recurring expenses, and pending credits.

Pending credits are not treated as spendable cash.

### 3. 90-day simulation

The engine projects the balance day by day.

A payment is safe only when:

    projected balance >= minimum balance to keep

The same simulation is used to evaluate candidate payment schedules and eligible spending changes.

### 4. Candidate generation

The agent considers:

- full payment
- supplied installment plans
- partial payment
- waiting until a safe date
- reducing or stopping eligible flexible expenses

### 5. Deterministic selection

Feasible candidates are selected using explicit deterministic criteria:

1. completion within the requested deadline
2. fewer spending changes
3. lower total payable amount
4. earlier payment timing
5. fewer payments
6. stable option ordering

No LLM is used to make the final financial choice.

### 6. Independent verification

Before serialization, the verifier checks output invariants such as:

- amount bounds
- valid affordability status
- valid payment method
- payment-plan consistency
- partial-payment constraints
- deadline/date consistency
- required output schema

---

## Financial safety model

The financial engine is the source of truth for affordability.

### Minimum balance

The projected balance must remain at or above the user's configured minimum balance.

### Confirmed income

Pending credits are excluded from spendable cash calculations.

### Unknown debit amounts

If a debit amount is missing from structured data and cannot be recovered from image evidence, it is handled conservatively rather than silently becoming zero.

### Currency conversion

Events in a different currency are converted using the dated exchange-rate data.

### Flexible spending

Eligible recurring expenses can be reduced or stopped only when the user's profile permits the change. Combinatorial search is bounded to keep runtime predictable.

---

## AI and multi-agent layer

The repository contains an optional local Ollama-based semantic layer with specialized roles:

- **Request Understanding Agent** — extracts intent, urgency, and entities.
- **Evidence Agent** — interprets messages and OCR-derived evidence.
- **Strategy Reasoning Agent** — reasons over already-computed candidates.
- **Explanation Agent** — explains verified results using supplied facts.

The AI layer is deliberately non-authoritative:

> **It cannot calculate affordability, override the financial engine, change balances, approve a payment, or bypass verification.**

Messages and OCR results are treated as untrusted evidence, helping isolate prompt-injection content from financial decisions.

### Competition mode

The default competition path is deterministic and makes zero LLM calls. Ollama is optional for local demonstrations and semantic-reasoning experiments.

---

## ML risk layer

The project includes an IsolationForest-based anomaly/risk component.

Its purpose is to provide an additional ML signal from profile data. It does not determine affordability and cannot override hard financial constraints.

---

## Verification

The verifier is intentionally separate from the decision engine:

    Financial Engine
          |
          v
    "Here is the calculated plan."
          |
          v
       Verifier
          |
          v
    "Does it satisfy the required invariants?"
          |
          v
       output.csv

This separation provides an independent output-level safety check.

---

## Project structure

    hackerrank-orchestrate-september26/
    |
    +-- app/
    |   +-- agent.py              Main financial decision orchestration
    |   +-- ai_layer.py           Optional Ollama provider adapter
    |   +-- data_loader.py        Dataset loading and normalization
    |   +-- financial_engine.py   Cash-flow simulation and payment logic
    |   +-- image_layer.py        Image/OCR evidence handling
    |   +-- multi_agent.py        Optional specialized AI reasoning agents
    |   +-- risk_model.py         ML anomaly-risk component
    |   +-- verifier.py           Independent output validation
    |
    +-- code/
    |   +-- main.py               Competition entry point
    |
    +-- docs/
    |   +-- AGENT_ARCHITECTURE.md Detailed architecture notes
    |
    +-- evaluation/
    |   +-- README.md             Evaluation and usage documentation
    |
    +-- requests.csv
    +-- requirements.txt
    +-- .gitignore
    +-- README.md
    +-- output.csv                Generated after a run

The official challenge dataset is treated as runtime input rather than duplicated into the source tree.

---

## Input data

The runner expects the official challenge dataset directory in the project root.

The implementation consumes the challenge's structured sources, including:

- requests
- profiles
- financial events
- messages
- images/media
- payment options
- exchange rates

Place the supplied dataset according to the challenge's expected directory structure before running.

---

## Output

The runner produces output.csv with exactly these columns, in this order:

    request_id
    amount_safe_to_pay
    affordability_status
    recommended_payment_method
    payment_plan
    earliest_date_for_full_payment
    spending_changes_needed
    decision_explanation

### Affordability statuses

- affordable_now
- affordable_with_plan
- affordable_later
- not_affordable

### Payment methods

- full_payment
- partial_payment
- installments
- wait
- not_recommended

---

## Setup

### Requirements

- Python 3.11+
- pip
- official challenge dataset
- optional Ollama for local LLM experiments

Create a virtual environment and install dependencies:

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt

---

## Run

From the repository root:

    python code/main.py

The runner processes every request, prints progress, validates generated rows, writes output.csv, and creates evaluation/usage_report.md.

---

## Optional Ollama mode

The competition path does not require a cloud API.

Enable local Ollama reasoning with:

    $env:USE_OLLAMA="1"
    $env:OLLAMA_MODEL="qwen2.5:7b-instruct"
    python code/main.py

Multiple models can be configured:

    $env:AGENT_MODELS="qwen2.5:7b-instruct,llama3:latest,phi3:latest,gemma:2b"

Default endpoint:

    http://localhost:11434

No API key is required for local Ollama execution.

---

## Performance and reliability

The implementation includes:

- financial events indexed by user
- payment options indexed by request
- cached event amounts
- cached 90-day forecasts
- cached payment simulations
- direct safe-amount calculation instead of repeated binary search
- bounded flexible-spending combinations
- no mandatory per-request LLM explanation call
- deterministic competition path

These optimizations keep repeated state-building and simulation work out of the hot path.

---

## Evaluation artifacts

After a run, inspect:

    evaluation/usage_report.md

The report records:

- request count
- runtime
- LLM call count
- token usage
- provider/model information when applicable
- verification issue count
- runtime optimization notes
- safety boundaries

For a deterministic competition run, the expected LLM call count is zero.

---

## Design trade-offs

### Why not let an LLM make the final decision?

Affordability contains hard numerical constraints. An LLM can misunderstand numbers, invent schedules, overlook future expenses, treat pending income as available, or produce inconsistent arithmetic.

The system therefore uses AI where language understanding is useful and deterministic code where correctness is mandatory.

### Why a 90-day horizon?

A fixed forecast horizon provides a predictable computational boundary while covering scheduled future events relevant to the decision.

### Why include ML risk scoring?

The anomaly model demonstrates an additional ML signal without allowing a probabilistic component to override hard financial constraints.

### Why a separate verifier?

The decision engine and verifier have different responsibilities. Independent validation catches output-level inconsistencies before they reach the final CSV.

---

## Security and safety

- No API credentials are required for the default path.
- Local Ollama is optional.
- LLM output is not an authoritative financial decision.
- Messages and OCR results are treated as untrusted input.
- Minimum-balance constraints are enforced deterministically.
- Pending credits are not treated as confirmed spendable funds.
- Do not submit .venv, credentials, or private API keys.

---

## Submission checklist

- [ ] Place the official final dataset in the expected location.
- [ ] Create and activate the Python virtual environment.
- [ ] Install requirements.txt.
- [ ] Run python code/main.py.
- [ ] Confirm output.csv is generated.
- [ ] Confirm the required columns and ordering.
- [ ] Check the request count.
- [ ] Review evaluation/usage_report.md.
- [ ] Confirm no credentials or API keys are included.
- [ ] Exclude .venv and other local-only files.
- [ ] Package the required files according to the HackerRank submission instructions.

---

## Documentation

- [Agent Architecture](docs/AGENT_ARCHITECTURE.md)
- [Evaluation Notes](evaluation/README.md)

---

## Project

**FinAgent-AI**  
HackerRank Orchestrate — September 2026  
Challenge: **Buy or Wait?**
