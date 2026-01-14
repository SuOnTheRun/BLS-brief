import io
import math
import streamlit as st
import pandas as pd

from src.io import read_uploaded_file, validate_input, take_only_inputs
from src.metrics import compute_metrics
from src.insights import build_insight_cards
from src.pdf_report import build_pdf_bytes
from src.config import ALLOWED_INPUT_COLS
from src.charts import (
    interactive_lift_histogram,     # currently mapped to reliability mix
    interactive_confidence_scatter, # currently mapped to impact matrix
    interactive_lift_rank,          # currently mapped to forest plot
    interactive_dumbbell,           # single KPI story chart
    interactive_ci_interval         # same story chart (compat)
)

# -----------------------------
# Style (clean, leadership-safe)
# -----------------------------
CSS = """
<style>
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1220px; }
h1, h2, h3 { letter-spacing: -0.02em; }
.small-muted { color: rgba(49, 51, 63, 0.65); font-size: 0.92rem; line-height: 1.35; }

.hero {
  display: flex; gap: 16px; align-items: flex-start; justify-content: space-between;
  margin: 10px 0 6px 0;
}
.hero-left { flex: 1; }
.hero-right { width: 260px; display:flex; justify-content:flex-end; }

.card {
  border: 1px solid rgba(49, 51, 63, 0.12);
  border-radius: 14px;
  padding: 14px 16px;
  background: white;
}
.card-title { font-size: 0.85rem; color: rgba(49, 51, 63, 0.65); margin-bottom: 6px; }
.card-value { font-size: 1.25rem; font-weight: 760; margin: 0; }

.hr { height: 1px; background: rgba(49, 51, 63, 0.10); margin: 14px 0; }

.pill {
  display: inline-block; padding: 4px 10px; border-radius: 999px;
  border: 1px solid rgba(49, 51, 63, 0.15);
  font-size: 0.8rem; color: rgba(49, 51, 63, 0.75);
  background: rgba(49, 51, 63, 0.03);
}

.kpi-row { border: 1px solid rgba(49,51,63,0.10); border-radius: 14px; padding: 14px 16px; background: white; }
.kpi-head { font-weight: 760; font-size: 1.0rem; margin-bottom: 6px; }
.kpi-sub { color: rgba(49,51,63,0.65); font-size: 0.85rem; margin-bottom: 10px; }
.note { color: rgba(49,51,63,0.85); font-size: 0.95rem; line-height: 1.45; }

.takeaway {
  border: 1px solid rgba(49,51,63,0.12);
  background: rgba(49,51,63,0.02);
  border-radius: 14px;
  padding: 12px 14px;
}
.takeaway-title { font-weight: 760; margin-bottom: 6px; }
.takeaway-body { color: rgba(49,51,63,0.78); font-size: 0.92rem; line-height: 1.35; }

.defline { color: rgba(49,51,63,0.78); font-size: 0.92rem; line-height: 1.35; }
.defline b { color: rgba(49,51,63,0.92); }
</style>
"""

st.set_page_config(page_title="BLS Brief", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

# -----------------------------
# Definitions (plain language)
# -----------------------------
SUMMARY_DEFS = {
    "Rows in view": "How many result lines you’re currently looking at, after filters. Each row is one KPI result for a brand / market / month.",
    "Statistically clear": "How many rows have a strong enough signal that we can treat the change as real, not likely random noise (95% confidence).",
    "Average lift": "Typical relative change vs control across the rows you’re viewing. Helpful for direction, but don’t treat it as a single ‘overall performance’ score.",
    "Average gap": "Typical absolute difference between exposed and control, in percentage points. This is the ‘real world’ size of the change."
}

RELIABILITY_DEFS = {
    "High": "Clear result + healthy sample sizes. Safe to use as a conclusion.",
    "Medium": "Clear result, but sample is not ideal. Still useful, but be cautious.",
    "Directional": "Not statistically clear, but shows a pattern worth watching. Treat as a signal, not a conclusion.",
    "Low": "Too little data (or too noisy). Avoid using for decisions."
}

IMPACT_MATRIX_DEFS = [
    ("Act", "Positive and statistically clear. Prioritise these for messaging and investment."),
    ("Watch", "Positive but not clear. Don’t claim it as a win yet—monitor or test more."),
    ("Investigate", "Negative and clear. Treat as a risk or friction to address."),
    ("Ignore", "Negative but not clear. Avoid overreacting—could be noise.")
]

def _template_csv_bytes() -> bytes:
    df = pd.DataFrame(columns=ALLOWED_INPUT_COLS)
    bio = io.StringIO()
    df.to_csv(bio, index=False)
    return bio.getvalue().encode("utf-8")

def _fmt_pct(x):
    try:
        return f"{float(x):.2f}%"
    except:
        return "—"

def _fmt_pts(x):
    try:
        return f"{float(x):.2f} pts"
    except:
        return "—"

def _fmt_p(x):
    try:
        v = float(x)
        if v < 0.0001:
            return "<0.0001"
        return f"{v:.4f}"
    except:
        return "—"

def _p_explain(p):
    """
    A simple translation of p-value into plain English, without jargon.
    """
    try:
        p = float(p)
    except:
        return "We couldn’t compute certainty for this row."
    if p < 0.01:
        return "Very unlikely to be random noise. Strong confidence."
    if p < 0.05:
        return "Unlikely to be random noise. Good confidence."
    if p < 0.10:
        return "Could be real, could be noise. Treat as directional."
    return "Quite likely to be noise. Don’t use as a conclusion."

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.subheader("Inputs")

    strict_mode = st.checkbox(
        "Strict input mode",
        value=False,
        help="If on: upload is rejected when extra computed columns exist. If off: extra columns are ignored."
    )

    st.divider()
    st.subheader("Views")

    include_non_sig = st.checkbox(
        "Include non-definitive results",
        value=True,
        help="Keeps results that are not statistically definitive, with a note."
    )

    allow_exclude_non_sig = st.checkbox(
        "Allow removal of non-definitive results",
        value=True
    )

    compare_mode = st.checkbox(
        "Comparison view",
        value=True,
        help="Adds the impact matrix and the confidence interval ranking view."
    )

    show_definitions = st.checkbox(
        "Show definitions under sections",
        value=True
    )

    st.divider()
    st.subheader("PDF")
    report_title = st.text_input("Title", value="BLS Brief")
    pdf_scope = st.radio("Export scope", ["Selected row only", "All rows in view"], index=0)

# -----------------------------
# Header + fixed placement for template button
# -----------------------------
st.markdown("## BLS Brief")

st.markdown(
    """
<div class="hero">
  <div class="hero-left">
    <div class="small-muted">
      Upload inputs only. The platform calculates the stats (from scores + samples), shows interactive visuals, and exports a PDF.
    </div>
    <div class="small-muted" style="margin-top: 6px;">
      Template includes: base descriptors + Control Sample, Exposed Sample, Control Score, Exposed Score (optional: Study ID, KPI Order). No computed columns needed.
    </div>
  </div>
  <div class="hero-right">
  </div>
</div>
""",
    unsafe_allow_html=True
)

# Put the download button in the "hero-right" area by using columns
cH1, cH2 = st.columns([3, 1])
with cH2:
    st.download_button(
        "Download template",
        data=_template_csv_bytes(),
        file_name="BLS_input_template.csv",
        mime="text/csv",
        help="This template contains only the allowed input columns. The platform calculates everything else."
    )

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# -----------------------------
# Upload
# -----------------------------
uploaded = st.file_uploader("Upload CSV or XLSX", type=["csv", "xlsx", "xls"])
if uploaded is None:
    st.info("Upload a file to begin.")
    st.stop()

# Read
try:
    raw = read_uploaded_file(uploaded)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

# Validate + enforce input-only
check = validate_input(raw)

if not check["ok"]:
    st.error("Your upload is missing required input columns.")
    st.write("Missing base columns:", check["missing_base"])
    st.write("Missing score columns:", check["missing_scores"])
    st.write("Allowed inputs:", check["allowed_inputs"])
    st.stop()

extras = check.get("extras", [])
if extras:
    if strict_mode:
        st.error("This upload contains extra (computed) columns. Strict mode is on.")
        st.write("Remove these columns and upload again:", extras)
        st.write("Allowed inputs:", check["allowed_inputs"])
        st.stop()
    else:
        st.warning("This upload contains extra (computed) columns. They will be ignored by the platform.")
        st.write("Ignored columns:", extras)

inputs = take_only_inputs(raw)
df = compute_metrics(inputs)  # platform computes all stats

# -----------------------------
# Filters
# -----------------------------
st.markdown("### Filter")
c1, c2, c3, c4 = st.columns(4)
with c1:
    market = st.multiselect("Market", sorted(df["Market"].dropna().astype(str).unique().tolist()))
with c2:
    category = st.multiselect("Category", sorted(df["Category"].dropna().astype(str).unique().tolist()))
with c3:
    brand = st.multiselect("Brand", sorted(df["Brand"].dropna().astype(str).unique().tolist()))
with c4:
    kpi = st.multiselect("KPI", sorted(df["KPI"].dropna().astype(str).unique().tolist()))

filtered = df.copy()
if market:
    filtered = filtered[filtered["Market"].astype(str).isin(market)]
if category:
    filtered = filtered[filtered["Category"].astype(str).isin(category)]
if brand:
    filtered = filtered[filtered["Brand"].astype(str).isin(brand)]
if kpi:
    filtered = filtered[filtered["KPI"].astype(str).isin(kpi)]

# Optional removal of non-definitive
if allow_exclude_non_sig:
    remove_non_sig = st.checkbox("Remove non-definitive rows", value=False)
else:
    remove_non_sig = False

if remove_non_sig:
    filtered = filtered[filtered["Significant_95"] == True]
    include_non_sig_effective = False
else:
    include_non_sig_effective = include_non_sig

if len(filtered) == 0:
    st.warning("No rows match the current filters.")
    st.stop()

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# -----------------------------
# Summary (with definitions)
# -----------------------------
st.markdown("### Summary")

total = len(filtered)
clear_count = int(filtered["Significant_95"].sum())
avg_lift = float(filtered["Lift_Pct"].mean())
avg_abs_diff = float(filtered["Diff_PctPts"].abs().mean())

cc1, cc2, cc3, cc4 = st.columns(4)
with cc1:
    st.markdown(f'<div class="card"><div class="card-title">Rows in view</div><div class="card-value">{total}</div></div>', unsafe_allow_html=True)
with cc2:
    st.markdown(f'<div class="card"><div class="card-title">Statistically clear</div><div class="card-value">{clear_count}</div></div>', unsafe_allow_html=True)
with cc3:
    st.markdown(f'<div class="card"><div class="card-title">Average lift</div><div class="card-value">{avg_lift:.2f}%</div></div>', unsafe_allow_html=True)
with cc4:
    st.markdown(f'<div class="card"><div class="card-title">Average gap</div><div class="card-value">{avg_abs_diff:.2f} pts</div></div>', unsafe_allow_html=True)

if show_definitions:
    st.markdown(
        f"""
<div class="takeaway">
  <div class="takeaway-title">What these numbers mean</div>
  <div class="takeaway-body">
    <div class="defline"><b>Rows in view:</b> {SUMMARY_DEFS["Rows in view"]}</div>
    <div class="defline"><b>Statistically clear:</b> {SUMMARY_DEFS["Statistically clear"]}</div>
    <div class="defline"><b>Average lift:</b> {SUMMARY_DEFS["Average lift"]}</div>
    <div class="defline"><b>Average gap:</b> {SUMMARY_DEFS["Average gap"]}</div>
  </div>
</div>
""",
        unsafe_allow_html=True
    )

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# -----------------------------
# Deep dive moved UP (your request)
# -----------------------------
st.markdown("### Deep dive")

selector_df = filtered.reset_index(drop=True).copy()
selector_df["RowLabel"] = (
    selector_df["Brand"].astype(str)
    + " • " + selector_df["KPI"].astype(str)
    + " • " + selector_df["Month Year"].astype(str)
)

selected_label = st.selectbox("Choose a row", selector_df["RowLabel"].tolist(), index=0)
idx = selector_df.index[selector_df["RowLabel"] == selected_label][0]
row = selector_df.loc[idx].to_dict()

cards_all = build_insight_cards(selector_df, include_non_sig=include_non_sig_effective)
card = cards_all[idx]

meta = f"{row.get('Month Year','')} • {row.get('Category','')} • {row.get('Market','')}"
st.markdown(
    f"""
    <div class="kpi-row">
      <div class="kpi-head">{row.get('Brand','')} • {row.get('KPI','')}</div>
      <div class="kpi-sub">{meta} &nbsp; <span class="pill">{card["state_label"]}</span></div>
      <div class="note"><b>Note:</b> {card["note"]}</div>
      <div class="note"><b>What changed:</b> {card["meaning"]}</div>
      <div class="note"><b>How to use it:</b> {card["decision"]}</div>
    </div>
    """,
    unsafe_allow_html=True
)

# Stat explanation in plain language (so people don't get flabbergasted)
gap = row.get("Diff_PctPts", None)
lift = row.get("Lift_Pct", None)
pval = row.get("P_Value", None)
ci_lo = row.get("CI_Low_PctPts", None)
ci_hi = row.get("CI_High_PctPts", None)
rel = row.get("Reliability", "")

st.markdown(
    f"""
<div class="takeaway" style="margin-top: 10px;">
  <div class="takeaway-title">How to read the stats (plain language)</div>
  <div class="takeaway-body">
    <div class="defline"><b>Gap:</b> { _fmt_pts(gap) } — the exposed group is that many points higher/lower than control.</div>
    <div class="defline"><b>Lift:</b> { _fmt_pct(lift) } — the relative change vs control.</div>
    <div class="defline"><b>Certainty (p-value):</b> { _fmt_p(pval) } — {_p_explain(pval)}</div>
    <div class="defline"><b>Range (confidence interval):</b> [{_fmt_pts(ci_lo)}, {_fmt_pts(ci_hi)}] — the most likely window for the true change.</div>
    <div class="defline"><b>Reliability:</b> {rel} — a practical label that combines sample health + how clear the result is.</div>
  </div>
</div>
""",
    unsafe_allow_html=True
)

# Key numbers
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f'<div class="card"><div class="card-title">Control</div><div class="card-value">{row["Control_Pct"]:.2f}%</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="card"><div class="card-title">Exposed</div><div class="card-value">{row["Exposed_Pct"]:.2f}%</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="card"><div class="card-title">Gap</div><div class="card-value">{row["Diff_PctPts"]:.2f} pts</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown(f'<div class="card"><div class="card-title">Lift</div><div class="card-value">{row["Lift_Pct"]:.2f}%</div></div>', unsafe_allow_html=True)

# Story chart
st.plotly_chart(interactive_dumbbell(row), use_container_width=True)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# -----------------------------
# What moved most / least
# -----------------------------
st.markdown("### What moved most / least (quick scan)")

tmp = filtered.copy()
tmp["Label"] = tmp["Brand"].astype(str) + " • " + tmp["KPI"].astype(str)

top_pos = tmp.sort_values("Diff_PctPts", ascending=False).head(6)
top_neg = tmp.sort_values("Diff_PctPts", ascending=True).head(6)

cA, cB = st.columns(2)
with cA:
    st.markdown("**Top increases**")
    st.dataframe(top_pos[["Label","Diff_PctPts","Lift_Pct","P_Value","Reliability"]], use_container_width=True, height=240)
with cB:
    st.markdown("**Top declines**")
    st.dataframe(top_neg[["Label","Diff_PctPts","Lift_Pct","P_Value","Reliability"]], use_container_width=True, height=240)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# -----------------------------
# Overview visuals: Reliability mix + explanation + takeaway box
# -----------------------------
st.markdown("### Overview visuals")

st.plotly_chart(interactive_lift_histogram(filtered), use_container_width=True)

if show_definitions:
    st.markdown(
        f"""
<div class="takeaway" style="margin-top: 10px;">
  <div class="takeaway-title">How to read the reliability chart</div>
  <div class="takeaway-body">
    <div class="defline"><b>What this shows:</b> how many rows in your current view are safe to use as conclusions vs signals vs noise.</div>
    <div class="defline"><b>High:</b> {RELIABILITY_DEFS["High"]}</div>
    <div class="defline"><b>Medium:</b> {RELIABILITY_DEFS["Medium"]}</div>
    <div class="defline"><b>Directional:</b> {RELIABILITY_DEFS["Directional"]}</div>
    <div class="defline"><b>Low:</b> {RELIABILITY_DEFS["Low"]}</div>
  </div>
</div>
""",
        unsafe_allow_html=True
    )

# Short takeaway based on counts
counts = filtered["Reliability"].value_counts().to_dict()
high = int(counts.get("High", 0))
med = int(counts.get("Medium", 0))
dirn = int(counts.get("Directional", 0))
low = int(counts.get("Low", 0))

st.markdown(
    f"""
<div class="takeaway" style="margin-top: 10px;">
  <div class="takeaway-title">Quick takeaway</div>
  <div class="takeaway-body">
    In this view: <b>{high}</b> strong conclusions, <b>{med}</b> usable but watch sample size, <b>{dirn}</b> directional signals, <b>{low}</b> rows to ignore for decision-making.
    Use the <b>impact matrix</b> to prioritise what to act on.
  </div>
</div>
""",
    unsafe_allow_html=True
)

# -----------------------------
# Comparison view: Add explanation + takeaway box
# -----------------------------
if compare_mode and len(filtered) > 1:
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown("### Comparison view")

    st.markdown(
        """
<div class="takeaway">
  <div class="takeaway-title">What this section is for</div>
  <div class="takeaway-body">
    This view helps you prioritise. It separates “big changes” from “noisy changes” so you don’t chase randomness.
    Use the impact matrix first (action lens), then the confidence-interval ranking (evidence lens).
  </div>
</div>
""",
        unsafe_allow_html=True
    )

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # Impact Matrix
    st.plotly_chart(interactive_confidence_scatter(filtered), use_container_width=True)

    if show_definitions:
        bullets = "".join([f"<div class='defline'><b>{name}:</b> {desc}</div>" for name, desc in IMPACT_MATRIX_DEFS])
        st.markdown(
            f"""
<div class="takeaway" style="margin-top: 10px;">
  <div class="takeaway-title">How to read the impact matrix</div>
  <div class="takeaway-body">
    <div class="defline"><b>Dots:</b> each dot is one row (one KPI result).</div>
    <div class="defline"><b>X-axis (Lift):</b> right = positive change, left = negative change.</div>
    <div class="defline"><b>Y-axis (Certainty):</b> higher = stronger evidence the change is real.</div>
    <div class="defline"><b>Quadrants:</b></div>
    {bullets}
    <div class="defline" style="margin-top: 8px;">
      <b>How it’s calculated:</b> lift comes from exposed vs control scores. certainty comes from the p-value of the difference test.
    </div>
  </div>
</div>
""",
            unsafe_allow_html=True
        )

    # Evidence lens: forest plot
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.plotly_chart(interactive_lift_rank(filtered), use_container_width=True)

    st.markdown(
        """
<div class="takeaway" style="margin-top: 10px;">
  <div class="takeaway-title">How to read the confidence interval ranking</div>
  <div class="takeaway-body">
    Each line shows the most likely range for the true change (confidence interval). The dot is the best estimate.
    If the line crosses 0, the change may not be real. If it stays fully above or below 0, it’s stronger.
  </div>
</div>
""",
        unsafe_allow_html=True
    )

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# -----------------------------
# Results table (kept)
# -----------------------------
st.markdown("### Results table")

table_cols = [
    "Month Year", "Brand", "Category", "Market", "KPI",
    "Control Sample", "Exposed Sample",
    "Control_Pct", "Exposed_Pct",
    "Diff_PctPts", "Lift_Pct",
    "CI_Low_PctPts", "CI_High_PctPts",
    "Z_Score", "P_Value", "Significant_95",
    "Effect_Size_h", "Effect_Size_Qual", "Reliability",
    "Data_Flag"
]
existing = [c for c in table_cols if c in filtered.columns]
st.dataframe(filtered[existing].reset_index(drop=True), use_container_width=True, height=340)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# -----------------------------
# Export PDF
# -----------------------------
st.markdown("### Export")

if st.button("Generate PDF"):
    if pdf_scope == "Selected row only":
        pdf_df = selector_df.iloc[[idx]].reset_index(drop=True)
        pdf_cards = build_insight_cards(pdf_df, include_non_sig=include_non_sig_effective)
    else:
        pdf_df = filtered.reset_index(drop=True)
        pdf_cards = build_insight_cards(pdf_df, include_non_sig=include_non_sig_effective)

    pdf_bytes = build_pdf_bytes(
        pdf_df,
        pdf_cards,
        report_title=report_title,
        include_comparisons=compare_mode
    )

    st.download_button(
        "Download PDF",
        data=pdf_bytes,
        file_name="BLS_brief.pdf",
        mime="application/pdf"
    )
