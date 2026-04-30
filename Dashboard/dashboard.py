"""ProAg Track 2 — Streamlit Dashboard.

Internal advisor tool. Reads CSVs from CCDS/Outputs/ and shows three pages:
  1. Operation Overview — all cycles, headline metrics, P&L charts
  2. Cycle Detail       — drill into one Pig Group with guardrailed LLM summary
  3. Data Quality       — what the cleaning step did

Run from the CCDS folder:
    pip install streamlit pandas plotly anthropic
    streamlit run dashboard.py

Repo layout this file expects:
    CCDS/
    ├── Data/
    ├── Outputs/                    <- pipeline writes CSVs here
    │   ├── fact_cycle_pnl.csv
    │   ├── dim_cycle.csv
    │   ├── fact_cycle_costs.csv
    │   ├── fact_hedge_pnl.csv
    │   ├── fact_anomalies.csv
    │   ├── dq_report.csv
    │   └── llm_audit_log.jsonl     <- created by Cycle Detail page
    ├── proag_pipeline.ipynb
    ├── llm_guardrails.ipynb
    └── dashboard.py                <- this file
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

if Path("/workspaces/CCDS").exists():
    CCDS_ROOT = Path("/workspaces/CCDS")
else:
    CCDS_ROOT = Path(__file__).parent.resolve()

DATA_DIR    = CCDS_ROOT / "Data"
OUTPUT_DIR  = CCDS_ROOT / "Outputs"
AUDIT_LOG   = OUTPUT_DIR / "llm_audit_log.jsonl"

st.set_page_config(
    page_title="ProAg Producer Analytics",
    page_icon="🐖",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    if not OUTPUT_DIR.exists():
        return None
    needed = {
        "pnl":       "fact_cycle_pnl.csv",
        "cycles":    "dim_cycle.csv",
        "costs":     "fact_cycle_costs.csv",
        "hedge":     "fact_hedge_pnl.csv",
        "anomalies": "fact_anomalies.csv",
        "dq":        "dq_report.csv",
    }
    out, missing = {}, []
    for key, fname in needed.items():
        path = OUTPUT_DIR / fname
        if not path.exists():
            missing.append(fname)
        else:
            out[key] = pd.read_csv(path)
    if missing:
        return {"_missing": missing}
    return out


data = load_data()
if data is None:
    st.error(
        f"Outputs folder not found at `{OUTPUT_DIR}`.\n\n"
        "Run **proag_pipeline.ipynb** first, then refresh this page."
    )
    st.stop()
if "_missing" in data:
    st.error(
        f"Missing pipeline outputs in `{OUTPUT_DIR}`: "
        f"`{', '.join(data['_missing'])}`.\n\n"
        "Re-run **proag_pipeline.ipynb**."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Guardrailed LLM helpers (mirrors llm_guardrails.ipynb so dashboard is
# self-contained — no notebook import needed)
# ---------------------------------------------------------------------------

ANTHROPIC_MODEL = "claude-sonnet-4-20250514"


@dataclass
class TokenMap:
    forward: dict = field(default_factory=dict)
    reverse: dict = field(default_factory=dict)

    def add(self, real: str, kind: str) -> str:
        if real in self.forward:
            return self.forward[real]
        prefix = kind.upper()
        n = sum(1 for v in self.forward.values() if v.startswith(f"<{prefix}_"))
        token = f"<{prefix}_{chr(ord('A') + n)}>"
        self.forward[real] = token
        self.reverse[token] = real
        return token

    def anonymize(self, text: str) -> str:
        out = text
        for real in sorted(self.forward, key=len, reverse=True):
            out = re.sub(re.escape(real), self.forward[real], out)
        return out

    def restore(self, text: str) -> str:
        out = text
        for token, real in self.reverse.items():
            out = out.replace(token, real)
        return out


def build_token_map(producers=None, vendors=None) -> TokenMap:
    tm = TokenMap()
    for p in producers or []:
        if p:
            tm.add(p, "PRODUCER")
    for v in vendors or []:
        if v:
            tm.add(v, "VENDOR")
    return tm


PROHIBITED_PATTERNS = [
    (r"\b(predict|forecast|will|should)\b.*\b(price|hog|corn|cwt|market)\b",
     "Price prediction"),
    (r"\b(price|hog|cwt)\b.*\b(go up|go down|rise|fall|next week|next month)\b",
     "Price prediction"),
    (r"\b(should|recommend|suggest|advise)\b.*\b(hedge|sell|buy|cover|lock)\b",
     "Hedging recommendation"),
    (r"\b(when|how much|what)\b.*\b(should|to)\b.*\b(hedge|cover)\b",
     "Hedging recommendation"),
    (r"\bcompare\b.*\b(producer|operation|farm)\b", "Cross-producer comparison"),
    (r"\bother\s+producer", "Cross-producer comparison"),
]


def check_prohibited(message: str):
    text = message.lower()
    for pattern, label in PROHIBITED_PATTERNS:
        if re.search(pattern, text):
            return label
    return None


CITATION_RE = re.compile(r"\[\[(\w+)=([\w\-\d\.,]+)\]\]")


def verify_citations(text, metrics, tolerance=0.01):
    errors = []
    found = 0

    def _replace(m):
        nonlocal found
        found += 1
        key, cited = m.group(1), m.group(2)
        if key not in metrics:
            errors.append(f"Cited unknown key: {key}")
            return m.group(0)
        actual = metrics[key]
        if actual is None:
            errors.append(f"Cited {key} but actual is None")
            return m.group(0)
        if isinstance(actual, str):
            if cited != actual:
                errors.append(f"Mismatch for {key}: cited {cited!r}, actual {actual!r}")
                return m.group(0)
            return cited
        try:
            cited_num = float(cited.replace(",", ""))
        except ValueError:
            errors.append(f"Could not parse number for {key}: {cited!r}")
            return m.group(0)
        if abs(cited_num - float(actual)) > tolerance * max(1.0, abs(float(actual))):
            errors.append(f"Mismatch for {key}: cited {cited_num}, actual {actual}")
            return m.group(0)
        return cited

    clean = CITATION_RE.sub(_replace, text)
    return clean, (found > 0 and not errors), errors


def audit_log_event(event: dict) -> None:
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": time.time(), **event}, default=str) + "\n")


CYCLE_SUMMARY_PROMPT = """\
You are an analyst writing a one-paragraph briefing for a producer-advisor call.

Rules — non-negotiable:
1. Use ONLY the numbers in the metrics block below. Never invent or estimate.
2. Wrap every number you mention in citation tags: [[key=value]]
   Example: "received [[received_head=2399]] head"
   Use the exact key from the metrics block and the exact numeric value.
3. Do NOT predict prices, recommend trades, or speculate beyond the data.
4. Keep it to 3-4 sentences.
5. If a metric is null/missing, omit it. Do not guess.

Metrics:
{metrics}

Write the summary now.
"""


def _fallback_summary(metrics: dict) -> str:
    parts = [f"Cycle [[cycle_id={metrics['cycle_id']}]]"]
    if metrics.get("received_head") is not None:
        parts.append(f"received [[received_head={metrics['received_head']}]] head")
    if metrics.get("net_pnl") is not None:
        parts.append(f"and finished with $[[net_pnl={metrics['net_pnl']}]] net P&L")
    if metrics.get("pnl_per_head") is not None:
        parts.append(f"([[pnl_per_head={metrics['pnl_per_head']}]] per head)")
    return " ".join(parts) + "."


def call_llm(prompt: str, api_key: str | None) -> tuple[str | None, bool]:
    """Returns (response_text or None, used_fallback). None means fallback needed."""
    if not api_key:
        return None, True
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=ANTHROPIC_MODEL, max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            b.text for b in msg.content if getattr(b, "type", "") == "text"
        ).strip()
        return text, False
    except Exception as e:
        st.warning(f"LLM call failed: {e} — using fallback.")
        return None, True


def get_cycle_metrics(cycle_id: str) -> dict:
    row = data["pnl"][data["pnl"]["cycle_id"] == cycle_id]
    if row.empty:
        raise ValueError(f"Unknown cycle_id: {cycle_id}")
    r = row.iloc[0]
    return {
        "cycle_id": cycle_id,
        "received_head": int(r["received_head"]),
        "packer_revenue": round(float(r["packer_revenue"]), 2),
        "total_cost": round(float(r["total_cost"]), 2),
        "hedge_pnl": round(float(r["hedge_pnl"]), 2),
        "net_pnl": round(float(r["net_pnl"]), 2),
        "pnl_per_head": round(float(r["pnl_per_head"]), 2)
            if pd.notna(r["pnl_per_head"]) else None,
    }


def summarize_cycle_safely(user, cycle_id, user_request, producer_name, vendor_names, api_key):
    """Full guardrailed summary pipeline. Returns dict with status/summary."""
    base = {"user": user, "producer": cycle_id, "request": user_request}

    refusal = check_prohibited(user_request)
    if refusal:
        audit_log_event({**base, "event": "refused", "reason": refusal})
        return {"status": "refused", "refusal_reason": refusal, "summary": None}

    try:
        metrics = get_cycle_metrics(cycle_id)
    except ValueError as e:
        audit_log_event({**base, "event": "error", "detail": str(e)})
        return {"status": "error", "summary": None, "refusal_reason": str(e)}

    tm = build_token_map(producers=[producer_name] if producer_name else [], vendors=vendor_names or [])
    safe_metrics = json.loads(tm.anonymize(json.dumps(metrics)))
    prompt = CYCLE_SUMMARY_PROMPT.format(metrics=json.dumps(safe_metrics, indent=2))

    audit_log_event({
        **base, "event": "prompt_sent",
        "metrics_keys": list(metrics.keys()),
        "anonymized": list(tm.forward.values()),
    })

    raw, used_fallback = call_llm(prompt, api_key)
    if raw is None:
        raw = _fallback_summary(safe_metrics)
        used_fallback = True

    response = tm.restore(raw)
    clean, ok, errors = verify_citations(response, metrics)

    audit_log_event({
        **base, "event": "response",
        "used_fallback": used_fallback,
        "verification_ok": ok,
        "verification_errors": errors,
    })

    if not ok:
        return {
            "status": "blocked",
            "refusal_reason": "Citation verification failed",
            "verification_errors": errors,
            "raw_response": response,
            "metrics_sent": metrics,
        }
    return {
        "status": "ok",
        "summary": clean,
        "metrics_sent": metrics,
        "used_fallback": used_fallback,
    }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🐖 ProAg")
    st.caption("Internal advisor tool — Track 2")

    page = st.radio(
        "Page",
        ["Operation Overview", "Cycle Detail", "Data Quality"],
        label_visibility="collapsed",
    )

    st.divider()
    st.subheader("Pipeline status")
    st.write(f"Cycles tracked: **{len(data['pnl'])}**")
    st.write(f"Hedge records: **{len(data['hedge'])}**")
    st.write(f"Cost rows: **{len(data['costs'])}**")
    st.write(f"Anomalies: **{len(data['anomalies'])}**")

    st.divider()
    st.caption(f"📁 `{OUTPUT_DIR}`")
    if st.button("Reload data", use_container_width=True):
        load_data.clear()
        st.rerun()


# ---------------------------------------------------------------------------
# Page 1 — Operation Overview
# ---------------------------------------------------------------------------

def render_overview():
    st.title("Operation Overview")
    st.caption("All cycles for this producer.")

    pnl = data["pnl"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cycles tracked", f"{len(pnl)}")
    c2.metric("Total net P&L", f"${pnl['net_pnl'].sum():,.0f}")
    avg_pp = pnl["pnl_per_head"].mean()
    c3.metric("Avg P&L / head", f"${avg_pp:,.2f}" if pd.notna(avg_pp) else "—")
    losing = (pnl["net_pnl"] < 0).sum()
    c4.metric(
        "Cycles in the red",
        f"{losing} / {len(pnl)}",
        delta=f"{losing/max(len(pnl),1)*100:.0f}%",
        delta_color="inverse",
    )

    st.divider()

    st.subheader("Net P&L per Head by Cycle")
    df_sorted = pnl.dropna(subset=["pnl_per_head"]).sort_values("pnl_per_head")
    if df_sorted.empty:
        st.info("No P&L per head data available.")
    else:
        fig = px.bar(
            df_sorted, x="cycle_id", y="pnl_per_head",
            color="pnl_per_head", color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            labels={"pnl_per_head": "USD per head", "cycle_id": "Cycle"},
        )
        fig.update_layout(height=380, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Revenue vs. Cost")
        fig = px.scatter(
            pnl, x="total_cost", y="packer_revenue",
            size="received_head", hover_name="cycle_id",
            hover_data={"net_pnl": ":,.0f"},
            labels={"total_cost": "Allocated cost ($)",
                    "packer_revenue": "Packer revenue ($)"},
        )
        max_v = max(pnl["total_cost"].max(), pnl["packer_revenue"].max(), 1)
        fig.add_trace(go.Scatter(
            x=[0, max_v], y=[0, max_v], mode="lines",
            name="Break-even", line=dict(dash="dash", color="grey"),
        ))
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("P&L Composition (all cycles)")
        composition = pd.DataFrame({
            "component": ["Packer revenue", "Hedge P&L", "Allocated cost"],
            "amount":    [pnl["packer_revenue"].sum(),
                          pnl["hedge_pnl"].sum(),
                          -pnl["total_cost"].sum()],
        })
        fig = px.bar(
            composition, x="component", y="amount", color="component",
            color_discrete_map={
                "Packer revenue": "#2ca02c",
                "Hedge P&L": "#1f77b4",
                "Allocated cost": "#d62728",
            },
        )
        fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("All Cycles — P&L Detail")
    st.dataframe(
        pnl.style.format({
            "packer_revenue": "${:,.2f}", "total_cost": "${:,.2f}",
            "hedge_pnl": "${:,.2f}", "net_pnl": "${:,.2f}",
            "pnl_per_head": "${:,.2f}",
        }),
        use_container_width=True, hide_index=True,
    )

    if not data["anomalies"].empty:
        st.subheader("⚠️ Flagged Cycles")
        st.dataframe(data["anomalies"], use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page 2 — Cycle Detail
# ---------------------------------------------------------------------------

def render_cycle_detail():
    st.title("Cycle Detail")
    st.caption("Drill into one Pig Group. Use during a producer call.")

    cycle_id = st.selectbox("Select a cycle", sorted(data["pnl"]["cycle_id"].tolist()))
    metrics = get_cycle_metrics(cycle_id)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Head received", f"{metrics['received_head']:,}")
    c2.metric("Packer revenue", f"${metrics['packer_revenue']:,.0f}")
    c3.metric("Allocated cost", f"${metrics['total_cost']:,.0f}")
    c4.metric(
        "Net P&L", f"${metrics['net_pnl']:,.0f}",
        delta=(f"${metrics['pnl_per_head']:,.2f}/head"
               if metrics["pnl_per_head"] is not None else None),
    )

    st.divider()

    st.subheader("How this cycle got to its net P&L")
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=["Packer revenue", "Hedge P&L", "Cost", "Net P&L"],
        y=[metrics["packer_revenue"], metrics["hedge_pnl"],
           -metrics["total_cost"], metrics["net_pnl"]],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    fig.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cost breakdown")
    costs_for_cycle = data["costs"][data["costs"]["cycle_id"] == cycle_id]
    if costs_for_cycle.empty:
        st.info(
            "No costs allocated to this cycle. The dummy accounting file's "
            "Pig_Group_IDs do not match production cycles — real ProAg input "
            "needed to bridge this (open question §6 in the brief)."
        )
    else:
        agg = (costs_for_cycle
               .groupby(["cost_category", "confidence"], as_index=False)["allocated_cost"]
               .sum().sort_values("allocated_cost", ascending=False))
        st.dataframe(
            agg.style.format({"allocated_cost": "${:,.2f}"}),
            use_container_width=True, hide_index=True,
        )
        if "LOW" in set(costs_for_cycle["confidence"]):
            st.warning("Some costs at LOW confidence (Site-only fallback). Review with producer.")

    st.subheader("Hedging positions")
    hedges = data["hedge"][data["hedge"]["cycle_id"] == cycle_id]
    if hedges.empty:
        st.info("No hedging records for this cycle.")
    else:
        st.dataframe(
            hedges.style.format({
                "head_covered": "{:,}", "strike_cwt": "${:,.2f}",
                "settlement_cwt": "${:,.2f}", "hedge_gain_loss": "${:,.2f}",
            }),
            use_container_width=True, hide_index=True,
        )

    cycle_anomalies = data["anomalies"][data["anomalies"]["cycle_id"] == cycle_id]
    if not cycle_anomalies.empty:
        st.subheader("⚠️ Flags on this cycle")
        for _, a in cycle_anomalies.iterrows():
            st.warning(f"**{a['severity']}** — {a['note']} ({a['metric']} = {a['value']})")

    st.divider()

    # ---- LLM summary (guardrailed) ----
    st.subheader("Plain-English Summary")
    st.caption(
        "Generated with §5 guardrails: anonymization, computed-metrics-only, "
        "number verification, hard refusals, and audit logging."
    )

    with st.expander("Settings", expanded=False):
        api_key_default = os.environ.get("ANTHROPIC_API_KEY", "")
        api_key = st.text_input(
            "Anthropic API key (leave empty to use deterministic fallback)",
            value=api_key_default, type="password",
        )
        user_request = st.text_area(
            "Advisor's request",
            value="Summarize this cycle for the producer call.", height=70,
        )
        producer_name = st.text_input(
            "Producer name (anonymized before sending)", value="Demo Producer A",
        )
        vendor_names_raw = st.text_input(
            "Vendor names to anonymize (comma-separated)",
            value="VetHealth Inc, Iowa Feed Mill",
        )
        vendor_names = [v.strip() for v in vendor_names_raw.split(",") if v.strip()]

    if st.button("Generate summary", type="primary"):
        with st.spinner("Running through the guardrails..."):
            result = summarize_cycle_safely(
                user="streamlit_advisor", cycle_id=cycle_id,
                user_request=user_request, producer_name=producer_name,
                vendor_names=vendor_names, api_key=api_key or None,
            )

        if result["status"] == "ok":
            st.success(result["summary"])
            with st.expander("What was sent to the LLM (audit view)"):
                st.json(result["metrics_sent"])
                if result.get("used_fallback"):
                    st.caption(
                        "Used deterministic fallback (no API key). "
                        "Set ANTHROPIC_API_KEY to use Claude."
                    )
                else:
                    st.caption("Citations verified successfully.")
        elif result["status"] == "refused":
            st.error(f"Request refused: **{result['refusal_reason']}**")
            st.caption(
                "From §5 of the brief: hard refusals for price predictions, "
                "hedging recommendations, and cross-producer comparisons."
            )
        elif result["status"] == "blocked":
            st.error(f"Response blocked: {result['refusal_reason']}")
            with st.expander("Verification errors"):
                for err in result.get("verification_errors", []):
                    st.write(f"• {err}")
                st.caption("Raw response (not shown to user):")
                st.code(result.get("raw_response", ""))
        else:
            st.error(f"Error: {result.get('refusal_reason')}")


# ---------------------------------------------------------------------------
# Page 3 — Data Quality
# ---------------------------------------------------------------------------

def render_data_quality():
    st.title("Data Quality Report")
    st.caption(
        "Generated on every pipeline run. Shows null counts, dropped rows, "
        "imputations, and out-of-range sensor readings."
    )

    dq = data["dq"]

    row_summary = dq[dq["kind"] == "row_count"]
    if not row_summary.empty:
        st.subheader("Rows kept per table")
        st.dataframe(
            row_summary[["table", "count", "note"]].rename(
                columns={"count": "rows_kept", "note": "detail"}
            ),
            use_container_width=True, hide_index=True,
        )

    issues = dq[dq["kind"].isin(
        ["required_null", "duplicate", "out_of_range", "non_numeric",
         "imputed", "optional_null"]
    )]
    st.subheader("Cleaning actions taken")
    if issues.empty:
        st.success("No cleaning actions needed.")
    else:
        by_table = issues.groupby("table")["count"].sum().reset_index().rename(
            columns={"count": "total_actions"}
        )
        st.dataframe(by_table, use_container_width=True, hide_index=True)

        with st.expander("All cleaning actions (detail)"):
            st.dataframe(
                issues[["table", "kind", "column", "count", "action", "note"]],
                use_container_width=True, hide_index=True,
            )

    nulls = dq[dq["kind"] == "null_count"]
    if not nulls.empty:
        st.subheader("Null counts per column (before cleaning)")
        st.dataframe(
            nulls[["table", "column", "count", "note"]],
            use_container_width=True, hide_index=True,
        )

    # LLM audit log viewer
    st.divider()
    st.subheader("LLM audit log")
    if not AUDIT_LOG.exists():
        st.info("No LLM activity yet. Run a summary on the Cycle Detail page.")
    else:
        rows = []
        with AUDIT_LOG.open() as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if rows:
            audit_df = pd.DataFrame(rows)
            audit_df["timestamp"] = pd.to_datetime(audit_df["timestamp"], unit="s")
            st.dataframe(audit_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if page == "Operation Overview":
    render_overview()
elif page == "Cycle Detail":
    render_cycle_detail()
elif page == "Data Quality":
    render_data_quality()
