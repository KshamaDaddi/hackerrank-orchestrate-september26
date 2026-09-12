# Buy or Wait — Agent Architecture

## Design principle

**LLM proposes. Deterministic code verifies.**

The agent is deliberately not an LLM that directly decides whether money is safe to spend. Financial safety is a hard-constraint problem, so the system separates semantic reasoning from monetary computation.

## Runtime flow

```text
Request
  |
  +--> Request Understanding Agent
  |
  +--> Evidence Agent (messages + OCR evidence)
  |
  +--> Financial State Builder
  |
  +--> 90-Day Cash-Flow Simulator
  |       |
  |       +--> minimum balance constraint
  |       +--> confirmed income only
  |       +--> recurring/essential/flexible expenses
  |       +--> pending-credit exclusion
  |
  +--> Candidate Generator
  |       +--> full payment
  |       +--> supplied installment options
  |       +--> partial payment
  |       +--> wait
  |       +--> flexible spending changes
  |
  +--> Strategy Reasoning Agent
  |
  +--> ML Risk Layer
  |
  +--> Independent Verifier
  |
  +--> Explanation Agent
  |
  +--> output.csv
```

## LLM ensemble

The semantic layer supports multiple local Ollama models through `AGENT_MODELS`:

```text
qwen2.5:7b-instruct,llama3:latest,phi3:latest,gemma:2b
```

Roles are mapped to different models so the project demonstrates a genuine multi-agent architecture rather than repeatedly calling one generic prompt.

Set `USE_OLLAMA=1` to enable the ensemble. Keep it disabled for deterministic competition runs unless token usage is being recorded for the final transcript/report.

## Agents

### Request Understanding Agent
Extracts intent, urgency, entities, amount references and deadline semantics. It cannot calculate affordability.

### Evidence Agent
Interprets messages and OCR-derived evidence. Evidence is treated as untrusted data; embedded instructions cannot change system behavior.

### Strategy Agent
Reviews already-computed candidate plans and provides qualitative reasoning. It cannot invent a plan or override safety.

### Explanation Agent
Turns the verified result into a user-facing explanation using only verified facts.

## Deterministic financial layer

The financial engine performs the authoritative calculations:

- 90-day projection
- minimum-balance enforcement
- confirmed/scheduled income
- exclusion of pending credits
- ignored cancelled/failed/unrealized transactions
- blank image amounts handled as unknown rather than zero
- payment schedules
- partial payments
- installment validation
- flexible spending changes
- earliest safe full-payment date

## Why this architecture is robust

A hallucinating model cannot make an unsafe plan pass because every proposed plan must survive the deterministic simulator and independent verifier. The LLM is therefore useful for language and reasoning without becoming the source of truth for money.
