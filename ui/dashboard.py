from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app.agent import FinancialAgent

st.set_page_config(page_title="FinAgent AI", page_icon="◈", layout="wide")

st.markdown("""
<style>
.block-container { max-width: 1380px; padding-top: 2rem; }
.hero { padding: 1.2rem 1.4rem; border: 1px solid #263244; border-radius: 18px; background: linear-gradient(135deg,#111827,#0b1220); margin-bottom: 1rem; }
.hero h1 { margin:0; font-size:2.25rem; letter-spacing:-.04em; }
.hero p { color:#9aa8ba; margin:.45rem 0 0; }
.card { padding: 1rem 1.1rem; border:1px solid #263244; border-radius:14px; background:#111827; min-height:105px; }
.label { color:#8b98aa; font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; }
.value { font-size:1.55rem; font-weight:700; margin-top:.3rem; }
.safe { color:#58d68d; } .warn { color:#f5c451; } .danger { color:#ff7272; }
.small { color:#8b98aa; font-size:.86rem; }
.badge { display:inline-block; padding:.38rem .65rem; border-radius:999px; border:1px solid #344154; background:#182234; font-size:.8rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading financial engine…")
def load_agent():
    return FinancialAgent(ROOT / "dataset")

@st.cache_data
def run_all(_agent):
    return pd.DataFrame([_agent.decide(r) for _, r in _agent.data["requests"].iterrows()])

def money(value, currency):
    return f"{currency} {float(value):,.2f}"

def status_class(status):
    return "safe" if status in {"affordable_now", "affordable_with_plan"} else ("warn" if status == "affordable_later" else "danger")

st.markdown("""
<div class="hero">
  <div class="small">FINAGENT-AI · HACKERRANK ORCHESTRATE 2026</div>
  <h1>Buy or Wait?</h1>
  <p>Safety-first financial affordability intelligence — deterministic money engine, optional AI reasoning.</p>
</div>
""", unsafe_allow_html=True)

try:
    agent = load_agent()
except Exception as exc:
    st.error("Dataset not found or could not be loaded.")
    st.code(str(exc))
    st.info("Run this dashboard from the repository root after placing the official challenge dataset in ./dataset.")
    st.stop()

requests = agent.data["requests"]
profiles = agent.data["profiles"]

with st.sidebar:
    st.markdown("### ◈ FinAgent AI")
    st.caption("Financial decision intelligence")
    st.divider()
    request_ids = requests["request_id"].astype(str).tolist()
    selected_id = st.selectbox("Analyze request", request_ids)
    if st.button("Run full dataset", use_container_width=True):
        with st.spinner("Evaluating all requests…"):
            st.session_state["all_results"] = run_all(agent)
        st.success(f"Processed {len(requests)} requests.")
    st.divider()
    st.markdown("**Safety boundary**")
    st.caption("AI interprets evidence. Deterministic simulation owns balances, constraints and payment feasibility.")
    st.caption("No cloud API is required for the competition path.")

request = requests[requests["request_id"].astype(str) == selected_id].iloc[0]
result = agent.decide(request)
profile = agent.engine.profile(request.user_id)
currency = str(profile.get("home_currency", ""))
status = result["affordability_status"]
cls = status_class(status)

st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:end;margin:1rem 0 .8rem">
  <div><div class="small">REQUEST</div><h2 style="margin:.15rem 0 0">{selected_id}</h2></div>
  <div class="badge">{status.replace("_"," ").upper()}</div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="card"><div class="label">Requested</div><div class="value">{money(request.requested_amount, currency)}</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="card"><div class="label">Safe to pay today</div><div class="value {cls}">{money(result["amount_safe_to_pay"], currency)}</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="card"><div class="label">Current balance</div><div class="value">{money(profile.current_available_balance, currency)}</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="card"><div class="label">Minimum buffer</div><div class="value">{money(profile.minimum_balance_to_keep, currency)}</div></div>', unsafe_allow_html=True)

st.write("")
left, right = st.columns([1.25, 1])
with left:
    st.subheader("Decision")
    if status in {"affordable_now", "affordable_with_plan"}:
        st.success(result["decision_explanation"])
    elif status == "affordable_later":
        st.warning(result["decision_explanation"])
    else:
        st.error(result["decision_explanation"])
    d1, d2 = st.columns(2)
    d1.metric("Recommended method", result["recommended_payment_method"].replace("_", " ").title())
    d2.metric("Earliest full payment", result["earliest_date_for_full_payment"] or "Not within horizon")
    st.markdown("**Payment plan**")
    if result["payment_plan"] == "none":
        st.info("No safe payment plan.")
    else:
        rows = []
        for item in str(result["payment_plan"]).split("|"):
            date, amount = item.split(":", 1)
            rows.append({"Date": date, "Amount": money(amount, currency)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

with right:
    st.subheader("Financial state")
    st.dataframe(pd.DataFrame({
        "Metric": ["User", "Request date", "Completion deadline", "Home currency", "Payment methods"],
        "Value": [str(request.user_id), str(request.request_date), str(request.desired_completion_date), currency, str(profile.payment_methods_user_will_consider)]
    }), hide_index=True, use_container_width=True)
    st.markdown("**Flexible spending changes**")
    if result["spending_changes_needed"] == "none":
        st.caption("No spending changes required.")
    else:
        for change in result["spending_changes_needed"].split("|"):
            st.markdown(f"- {change}")

st.divider()
tab1, tab2, tab3 = st.tabs(["Cash-flow forecast", "Candidate audit", "Dataset overview"])

with tab1:
    dates, balances = agent.engine._forecast(request.user_id, request.request_date)
    chart = pd.DataFrame({"date": dates, "projected_balance": balances}).set_index("date")
    st.line_chart(chart, height=330)
    st.caption(f"90-day deterministic forecast · required minimum: {money(profile.minimum_balance_to_keep, currency)}")

with tab2:
    trace = agent.audit_traces.get(str(selected_id), {})
    a, b, c = st.columns(3)
    a.metric("Candidates evaluated", trace.get("candidate_count", 0))
    b.metric("Verified", "YES" if trace.get("verified") else "NO")
    c.metric("LLM calls", len(trace.get("llm_trace", [])))
    st.info("The verifier checks bounds, status/method consistency, payment-plan rules, deadline constraints and spending-change syntax before output serialization.")

with tab3:
    a, b, c = st.columns(3)
    a.metric("Requests", len(requests))
    b.metric("Profiles", len(profiles))
    c.metric("Financial events", len(agent.data["events"]))
    st.caption("Messages, OCR/image evidence, payment options and dated FX rates are incorporated by the underlying engine.")

st.divider()
st.caption("FinAgent-AI · HackerRank Orchestrate — September 2026 · AI interprets; deterministic code verifies.")
