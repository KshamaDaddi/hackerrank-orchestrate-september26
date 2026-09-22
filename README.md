# FinAgent-AI

> **Safety-first financial affordability agent built for HackerRank Orchestrate — September 2026**

FinAgent-AI is an AI-assisted financial decision system for the **“Buy or Wait?”** challenge. It evaluates whether a requested purchase can be paid safely now, supported by a payment plan, delayed until a safer date, or rejected under the available financial constraints.

The project is intentionally designed around a strict separation of responsibilities:

> **LLMs can interpret and explain. Deterministic code owns the money.**

That boundary keeps numerical affordability decisions reproducible, auditable, and independent of generative-model behavior.

---

## ✨ Highlights

- **90-day cash-flow forecasting**
- Financial-state reconstruction from structured and unstructured evidence
- Full, partial, installment, and wait strategies
- Deterministic payment-plan simulation
- Minimum-balance protection
- Confirmed-income handling
- Dated currency conversion
- Flexible-spending analysis with bounded search
- Independent output verification
- Optional local **Ollama** semantic/agent layer
- Optional **Isolation Forest** risk signal
- Interactive **Streamlit** dashboard
- Runtime optimizations through indexing and caching
- Competition path that can run **without LLM calls**

---

## 🎯 Problem

The “Buy or Wait?” problem is more than checking:

```
current_balance >= purchase_amount
```

A purchase may appear affordable today but become unsafe after future rent, bills, recurring expenses, or other committed transactions are considered.

FinAgent-AI therefore evaluates each request against the user's projected financial state and determines a payment strategy that respects hard constraints.

The system considers factors such as:

- Current balance
- Minimum balance to preserve
- Future income and expenses
- Pending and confirmed transactions
- Requested amount and deadline
- Available payment methods
- Installment options
- Partial-payment rules
- Eligible flexible expenses
- Financial information from messages/images
- Dated exchange rates

---

## 🧠 Core Design Principle

### AI handles interpretation. Deterministic code handles financial decisions.

The architecture deliberately avoids allowing an LLM to directly:

- approve a purchase
- calculate the authoritative balance
- modify financial events
- override a payment constraint
- bypass verification

Instead, AI is used where language understanding is useful, while the deterministic financial engine remains the source of truth.

This makes the system easier to reproduce, test, audit, and reason about.

---

## 🏗️ Architecture

```
                         Challenge Dataset
                                |
                                v
                         Data Normalization
                                |
                  +-------------+-------------+
                  |                           |
                  v                           v
          Financial State              Evidence / AI Layer
                  |                           |
                  v                           |
          90-Day Forecast                     |
                  |                           |
                  +-------------+-------------+
                                |
                                v
                     Candidate Generation
                                |
                                v
                    Deterministic Simulation
                                |
                                v
                     Strategy Selection
                                |
                                v
                    Independent Verifier
                                |
                                v
                           output.csv
```

### Safety boundary

```
        AI / ML Layer
             |
             +--> interpret evidence
             +--> extract semantics
             +--> generate optional explanations
             |
             v
   Deterministic Financial Engine
             |
             +--> balances
             +--> dates
             +--> cash-flow simulation
             +--> payment schedules
             +--> hard constraints
             +--> candidate feasibility
             |
             v
     Independent Verifier
             |
             v
          Final Output
```

---

## 🔄 Decision Pipeline

### 1. Financial State Reconstruction

Relevant profile information and financial events are combined into a normalized user state.

The state includes information such as:

- available balance
- minimum required balance
- currency
- financial events
- payment options
- recurring expenses
- eligible flexible spending

### 2. Event Normalization

Financial events are indexed by user and date.

The pipeline handles:

- credits and debits
- recurring transactions
- pending and confirmed transactions
- dated currency conversion
- missing/uncertain transaction evidence
- image/OCR-derived financial information

**Pending income is not treated as confirmed spendable cash.**

### 3. 90-Day Cash-Flow Forecast

The engine projects the user's balance over a fixed 90-day horizon.

A candidate payment is considered safe only when:

```
projected_balance >= minimum_required_balance
```

The same simulation framework is used when evaluating alternative payment schedules and eligible spending changes.

### 4. Candidate Generation

The system generates feasible alternatives such as:

- Full payment
- Supplied installment plans
- Partial payment
- Waiting until a safer date
- Eligible flexible-spending reductions

### 5. Deterministic Strategy Selection

Candidates are evaluated using explicit rules rather than an LLM-generated preference.

The selection logic considers:

1. Ability to complete within the requested deadline
2. Number of required spending changes
3. Total payable amount
4. Payment timing
5. Number of payments
6. Stable deterministic ordering

### 6. Independent Verification

Before writing the final CSV, a separate verifier checks output invariants including:

- amount bounds
- affordability status
- payment method
- payment-plan consistency
- partial-payment rules
- date/deadline consistency
- required schema

---

## 🤖 AI / Multi-Agent Layer

The repository includes an optional local Ollama-based semantic layer.

Specialized components can provide:

| Component | Responsibility |
|---|---|
| Request Understanding | Extract intent, urgency, entities and request semantics |
| Evidence Agent | Interpret messages and OCR-derived evidence |
| Strategy Reasoning | Reason over already-computed candidate plans |
| Explanation Agent | Generate explanations from verified facts |

The AI layer is **non-authoritative**.

It cannot calculate the authoritative affordability result, modify the financial state, approve a payment, or bypass the verifier.

### Competition Mode

The default competition pipeline is deterministic and does not require an LLM.

Ollama is optional and intended for local demonstrations and semantic-reasoning experiments.

---

## 📊 ML Risk Layer

FinAgent-AI also contains an **Isolation Forest** anomaly/risk component.

Its role is to provide an additional ML-derived signal from financial/profile features.

It does **not** determine affordability and cannot override the hard financial constraints enforced by the financial engine.

---

## 🛡️ Financial Safety Model

### Minimum Balance Protection

Projected balance must remain at or above the configured minimum balance throughout the evaluated payment schedule.

### Confirmed Income

Pending credits are excluded from spendable-cash calculations.

### Uncertain Debit Amounts

When a debit amount cannot be reliably recovered from structured or image evidence, the system handles the uncertainty conservatively rather than silently treating it as zero.

### Currency Conversion

Cross-currency financial events are converted using the available dated exchange-rate information.

### Flexible Spending

Only eligible flexible expenses can be modified, and the search is bounded to keep runtime predictable.

---

## 🧪 Verification Strategy

The verifier is intentionally separated from the decision engine.

```
Financial Engine
      |
      v
Calculated Candidate
      |
      v
Independent Verifier
      |
      +---- invalid ----> reject / flag
      |
      v
   Valid Output
      |
      v
  output.csv
```

This creates an output-level safety boundary rather than relying solely on the component that generated the decision.

---

## 🖥️ Interactive Dashboard

FinAgent-AI includes an optional Streamlit dashboard for exploring the same financial decision engine used by the competition entry point.

The dashboard provides:

- Request-level affordability analysis
- Safe-to-pay amount
- Recommended payment strategy
- Payment-plan details
- Current financial state
- 90-day cash-flow visualization
- Flexible-spending changes
- Candidate/audit information
- Dataset-level metrics

### Run the UI

```bash
pip install -r requirements-ui.txt
streamlit run ui/dashboard.py
```

The dashboard is a **demonstration layer**. It does not replace or modify `code/main.py`.

---

## 📁 Project Structure

```
hackerrank-orchestrate-september26/
│
├── app/
│   ├── agent.py
│   ├── ai_layer.py
│   ├── data_loader.py
│   ├── financial_engine.py
│   ├── image_layer.py
│   ├── multi_agent.py
│   ├── risk_model.py
│   └── verifier.py
│
├── code/
│   └── main.py
│
├── docs/
│   └── AGENT_ARCHITECTURE.md
│
├── evaluation/
│   └── README.md
│
├── ui/
│   └── dashboard.py
│
├── requests.csv
├── requirements.txt
├── requirements-ui.txt
├── .gitignore
└── README.md
```

### Key modules

| Module | Purpose |
|---|---|
| `app/agent.py` | Main financial decision orchestration |
| `app/financial_engine.py` | Forecasting, simulation and payment logic |
| `app/data_loader.py` | Dataset loading and normalization |
| `app/verifier.py` | Independent output validation |
| `app/ai_layer.py` | Optional Ollama provider integration |
| `app/multi_agent.py` | Optional specialized AI reasoning |
| `app/risk_model.py` | Isolation Forest risk/anomaly signal |
| `app/image_layer.py` | Image/OCR evidence processing |
| `code/main.py` | Competition entry point |
| `ui/dashboard.py` | Interactive Streamlit interface |

---

## 📥 Input Data

The runner expects the official challenge dataset in the required project structure.

The system consumes structured and supporting sources including:

- Requests
- User profiles
- Financial events
- Messages
- Images/media
- Payment options
- Exchange rates

Place the supplied challenge data in the expected location before running the competition pipeline.

---

## 📤 Output

The main runner generates:

`output.csv`

with the required columns:

```
request_id
amount_safe_to_pay
affordability_status
recommended_payment_method
payment_plan
earliest_date_for_full_payment
spending_changes_needed
decision_explanation
```

### Affordability statuses

- `affordable_now`
- `affordable_with_plan`
- `affordable_later`
- `not_affordable`

### Payment strategies

- `full_payment`
- `partial_payment`
- `installments`
- `wait`
- `not_recommended`

---

## ⚙️ Setup

### Requirements

- Python 3.11+
- pip
- Official challenge dataset
- Optional: Ollama for local LLM experiments

### Create environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
pip install -r requirements.txt
```

---

## ▶️ Run the Competition Pipeline

From the repository root:

```powershell
python code/main.py
```

The pipeline processes the requests, performs financial simulations, validates generated rows, writes `output.csv`, and produces evaluation information.

---

## 🦙 Optional Ollama Mode

The competition pipeline does not require a cloud API.

For local semantic-reasoning experiments:

```powershell
$env:USE_OLLAMA="1"
$env:OLLAMA_MODEL="qwen2.5:7b-instruct"
python code/main.py
```

Multiple local models can be configured:

```powershell
$env:AGENT_MODELS="qwen2.5:7b-instruct,llama3:latest,phi3:latest,gemma:2b"
```

Default Ollama endpoint:

```
http://localhost:11434
```

No cloud API key is required for local Ollama execution.

---

## ⚡ Performance Engineering

The implementation includes several optimizations designed to reduce repeated computation:

- Financial events indexed by user
- Payment options indexed by request
- Cached event amounts
- Cached 90-day forecasts
- Cached payment simulations
- Direct safe-amount calculation
- Bounded flexible-spending combinations
- No mandatory LLM explanation call
- Deterministic competition execution path

The goal is predictable runtime while preserving the financial constraints of the decision process.

---

## 📈 Evaluation Artifacts

After execution, inspect:

```
evaluation/usage_report.md
```

The report can capture:

- Request count
- Runtime
- LLM call count
- Token usage
- Provider/model information
- Verification issues
- Runtime optimization notes
- Safety boundaries

For the deterministic competition path, the expected LLM call count is **zero**.

---

## 🔍 Design Trade-offs

### Why not let an LLM make the final financial decision?

Financial affordability contains hard numerical constraints. Generative models can misread values, overlook future events, invent schedules, or produce inconsistent arithmetic.

The system therefore assigns language-heavy tasks to AI and numerical financial decisions to deterministic code.

### Why a 90-day horizon?

A fixed forecast horizon creates a predictable computational boundary while covering the scheduled future events relevant to the challenge.

### Why include an ML risk model?

The risk model demonstrates how an additional ML signal can be incorporated without allowing a probabilistic model to override deterministic financial constraints.

### Why use a separate verifier?

The decision engine and verifier have different responsibilities. Independent output validation provides another layer for catching schema, date, amount, and payment-plan inconsistencies.

---

## 🔐 Security & Safety

- Default execution does not require cloud API credentials.
- Ollama is optional.
- LLM output is not an authoritative financial decision.
- Messages and OCR results are treated as untrusted evidence.
- Minimum-balance constraints are enforced by deterministic logic.
- Pending income is not treated as confirmed spendable funds.
- Do not commit API keys, credentials, or local virtual environments.

---

## ✅ Reproducibility Checklist

Before running or submitting:

- [ ] Use Python 3.11+
- [ ] Activate the virtual environment
- [ ] Install `requirements.txt`
- [ ] Place the challenge dataset correctly
- [ ] Run `python code/main.py`
- [ ] Confirm `output.csv` is generated
- [ ] Confirm required columns and ordering
- [ ] Review `evaluation/usage_report.md`
- [ ] Check for verification issues
- [ ] Remove credentials and local-only files
- [ ] Follow the official HackerRank submission requirements

---

## 📚 Documentation

- [Agent Architecture](docs/AGENT_ARCHITECTURE.md)
- [Evaluation Notes](evaluation/README.md)

---

## 🏁 Project Context

**Project:** FinAgent-AI  
**Event:** HackerRank Orchestrate — September 2026  
**Challenge:** Buy or Wait?  
**Primary focus:** AI Agents · Financial Reasoning · ML · Deterministic Simulation · Verification

---

## 👩‍💻 Author

**Kshama Daddi**

AI & Data Science | Machine Learning | Generative AI

[GitHub](https://github.com/KshamaDaddi) · [LinkedIn](https://www.linkedin.com/in/kshamadaddi)
