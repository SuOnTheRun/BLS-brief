import io
import pandas as pd
import streamlit as st

from src.io import read_uploaded_file, validate_input, take_only_inputs
from src.metrics import compute_all_metrics
from src.charts import (
    executive_reliability_ribbon,
    executive_impact_matrix,
    executive_forest_plot,
    executive_story_card_chart,
)
from src.pdf_report import build_pdf_bytes


st.set_page_config(page_title="BLS Brief", layout="wide")


CSS = """
<style>
.block-container {padding-top: 2.0rem; padding-bottom: 2.0rem; max-width: 1250px;}
h1, h2, h3 {letter-spacing: -0.02em;}
.takeaway {
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 14px;
  padding: 14px 16px;
  background: rgba(255,255,255,0.75);
}
.takeaway-title {font-weight: 650; margin-bottom: 6px;}
.takeaway-sub {opacity: 0.75; font-size: 0.92rem; margin-top: 6px;}
.small {font-size: 0.92rem; opacity: 0.85;}
.kpi-badges {display:flex; gap:10px; align-items:center; margin-top:6px;}
.badge {
  display:inline-block; padding: 3px 10px; border-radius: 999px;
  border: 1px solid rgba(0,0,0,0.10); background: rgba(0,0,0,0.02);
  font-size: 0.85rem;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# -----------------------------
# Helper: safe dataframe display
# -----------------------------
def show_df_with_available_cols(df, wanted_cols):
    cols = [c for c in wanted_cols if c in df.columns]
    st.dataframe(df[cols], use_container_width=True)


# -----------------------------
# Header row: title + template button
# -----------------------------
c1, c2 = st.columns([0.78, 0.22], vertical_alignment="center")
with c1:
    st.title("BLS Brief")
    st.write(
        "Upload inputs only. The platform calculates the stats (from scores + samples), shows interactive visuals, and exports a PDF."
    )
    st.caption(
        "Template includes: Month Year, Brand, Category, Market, KPI, KPI Order (optional), Control Sample, Exposed Sample, Control Score, Exposed Score. No computed columns needed."
    )

with c2:
    template_cols = [
        "Month Year", "Brand", "Category", "Market", "KPI", "KPI Order",
        "Control Sample", "Exposed Sample", "Control Score", "Exposed Score"
    ]
    template_df = pd.DataFrame([{c: "" for c in template_cols}])
    buf = io.BytesIO()
    template_df.to_csv(buf, index=False)
    st.download_button(
        "Download template",
        data=buf.getvalue(),
        file_name="BLS_Brief_Template.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.divider()


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Inputs")
    strict_mode = st.checkbox(
        "Strict input mode",
        value=False,
        help="If on, rejects files with missing required columns instead of trying to interpret them."
    )

    st.header("Reliability settings (movable bands)")
    with st.expander("Edit thresholds", expanded=False):
        great_thr = st.slider("Great if min sample ≥", 100, 800, 300, step=10)
        good_thr = st.slider("Good if min sample ≥", 30, 400, 100, step=5)
        dir_thr = st.slider("Directional if min sample ≥", 10, 200, 50, step=5)
        st.caption(f"Current meaning: Great ≥ {great_thr}, Good ≥ {good_thr}, Directional ≥ {dir_thr}, otherwise Low")

    st.header("Views")
    include_non_def = st.checkbox("Include non-definitive results", value=True)
    allow_removal = st.checkbox("Allow removal of non-definitive results", value=True)
    show_comparison = st.checkbox("Comparison view", value=True)
    show_definitions = st.checkbox("Show definitions under sections", value=True)

    st.header("PDF")
    pdf_title = st.text_input("Title", value="BLS Brief")
    export_scope = st.radio("Export scope", ["Selected row only", "All rows in view"], index=0)


# -----------------------------
# Upload
# -----------------------------
st.subheader("Upload CSV or XLSX")
uploaded = st.file_uploader("Upload CSV or XLSX", type=["csv", "xlsx", "xls"], label_visibility="collapsed")
if not uploaded:
    st.stop()

raw = read_uploaded_file(uploaded)

if strict_mode:
    validate_input(raw)

inputs = take_only_inputs(raw)

df = compute_all_metrics(
    inputs,
    GREAT_THRESHOLD=great_thr,
    GOOD_THRESHOLD=good_thr,
    DIRECTIONAL_THRESHOLD=dir_thr,
)

# -----------------------------
# Filters
# -----------------------------
st.subheader("Filter")
f1, f2, f3, f4 = st.columns(4)

def pick(col, container):
    if col not in df.columns:
        return None
    opts = sorted([x for x in df[col].dropna().unique()])
    return container.selectbox(col, ["(All)"] + opts)

market = pick("Market", f1)
category = pick("Category", f2)
brand = pick("Brand", f3)
kpi = pick("KPI", f4)

view = df.copy()
for col, val in [("Market", market), ("Category", category), ("Brand", brand), ("KPI", kpi)]:
    if val and val != "(All)" and col in view.columns:
        view = view[view[col] == val]

# Human feedback sentence
parts = []
if brand and brand != "(All)":
    parts.append(str(brand))
if market and market != "(All)":
    parts.append(f"({market})")
if category and category != "(All)":
    parts.append(f"Category: {category}")
if kpi and kpi != "(All)":
    parts.append(f"KPI: {kpi}")

st.caption(f"Showing {' • '.join(parts) if parts else 'All results'} ({len(view)} rows).")

# removal logic uses new bands (but will not crash if missing)
if allow_removal:
    remove_non_def = st.checkbox("Remove non-definitive rows", value=False)
    if remove_non_def and "Reliability_Band" in view.columns:
        view = view[~view["Reliability_Band"].isin(["Directional", "Low"])]

if not include_non_def and "Reliability_Band" in view.columns:
    view = view[~view["Reliability_Band"].isin(["Directional", "Low"])]

st.divider()


# -----------------------------
# Summary
# -----------------------------
st.subheader("Summary")
s1, s2, s3, s4 = st.columns(4)

rows_in_view = len(view)

# "Statistically clear" = clarity is Clear
stat_clear = int((view["Clarity_Band"] == "Clear").sum()) if "Clarity_Band" in view.columns else 0

# Avg lift: use visible lift only (Lift_Pct may be NaN when hidden)
avg_lift = float(view["Lift_Pct"].mean()) if ("Lift_Pct" in view.columns and rows_in_view) else 0.0
avg_gap = float(view["Diff_PctPts"].mean()) if ("Diff_PctPts" in view.columns and rows_in_view) else 0.0

s1.metric("Rows in view", f"{rows_in_view}")
s2.metric("Statistically clear", f"{stat_clear}")
s3.metric("Average lift", f"{avg_lift:.2f}%")
s4.metric("Average gap", f"{avg_gap:.2f} pts")

if show_definitions:
    st.markdown(
        """
<div class="takeaway">
  <div class="takeaway-title">What these mean (plain language)</div>
  <div><b>Rows in view:</b> how many KPI results you’re looking at after filters. Each row is one KPI result for a brand / market / month.</div>
  <div><b>Statistically clear:</b> how many rows look very likely real (not random noise), given the number of responses.</div>
  <div><b>Average lift:</b> typical relative change vs control. Useful for direction, not a single “overall score”.</div>
  <div><b>Average gap:</b> typical absolute difference between exposed and control in points. This is the most “real-world” way to read impact.</div>
</div>
        """,
        unsafe_allow_html=True,
    )

st.divider()


# -----------------------------
# Deep dive (moved up)
# -----------------------------
st.subheader("Deep dive")

if rows_in_view == 0:
    st.info("No rows match the current filters.")
    st.stop()

if "Label" not in view.columns:
    view = view.copy()
    if "Month Year" in view.columns:
        view["Label"] = view["Brand"].astype(str) + " • " + view["KPI"].astype(str) + " • " + view["Month Year"].astype(str)
    else:
        view["Label"] = view["Brand"].astype(str) + " • " + view["KPI"].astype(str)

choice = st.selectbox("Choose a row", view["Label"].tolist())
row = view[view["Label"] == choice].iloc[0]

brand_name = str(row.get("Brand", ""))
kpi_name = str(row.get("KPI", ""))
month = str(row.get("Month Year", ""))

gap = float(row.get("Diff_PctPts", 0.0))
pval = float(row.get("P_Value", 1.0))
rel_band = str(row.get("Reliability_Band", row.get("Reliability", "")))
clar_band = str(row.get("Clarity_Band", ""))
human_what = str(row.get("Human_WhatHappened", ""))
human_conf = str(row.get("Human_Confidence", ""))

lift_val = row.get("Lift_Pct", float("nan"))
lift_hidden = bool(row.get("Lift_Hidden", False))

tag = "Clear" if clar_band == "Clear" else "Directional" if clar_band == "Directional" else "Unclear"

st.markdown(
    f"""
<div class="takeaway">
  <div class="takeaway-title">{brand_name} • {kpi_name}</div>
  <div class="small">{month} • {str(row.get("Category",""))} • {str(row.get("Market",""))}</div>

  <div class="kpi-badges">
    <span class="badge">Reliability: {rel_band}</span>
    <span class="badge">Clarity: {tag}</span>
  </div>

  <br/>
  <div><b>What happened:</b> {human_what}</div>
  <div><b>How sure are we:</b> {human_conf}</div>
</div>
    """,
    unsafe_allow_html=True,
)

# Interactive deep dive chart
st.plotly_chart(executive_story_card_chart(row), use_container_width=True)

# Show a compact “numbers for adults” line
ctrl = float(row.get("Control_Pct", 0.0))
exp = float(row.get("Exposed_Pct", 0.0))
cn = int(float(row.get("Control Sample", 0))) if "Control Sample" in row else 0
en = int(float(row.get("Exposed Sample", 0))) if "Exposed Sample" in row else 0

lift_str = "— (hidden)" if lift_hidden else (f"{float(lift_val):.2f}%" if lift_val == lift_val else "—")
st.caption(
    f"Numbers line: Control {ctrl:.2f}% (n={cn}) • Exposed {exp:.2f}% (n={en}) • Gap {gap:.2f} pts • Lift {lift_str} • p={pval:.4f}"
)

# Notes block if any
notes_short = str(row.get("Notes_Short", "")).strip()
if notes_short:
    st.markdown(
        f"""
<div class="takeaway">
  <div class="takeaway-title">Notes</div>
  <div>{notes_short}</div>
</div>
        """,
        unsafe_allow_html=True
    )

st.divider()


# -----------------------------
# What moved most / least
# -----------------------------
st.subheader("What moved most / least (quick scan)")

top_inc = view.sort_values("Diff_PctPts", ascending=False).head(7)
top_dec = view.sort_values("Diff_PctPts", ascending=True).head(7)

left, right = st.columns(2)
with left:
    st.caption("Top increases")
    show_df_with_available_cols(
        top_inc,
        ["Label", "Diff_PctPts", "Lift_Pct", "P_Value", "Reliability_Band", "Clarity_Band", "Notes_Short"]
    )

with right:
    st.caption("Top declines")
    show_df_with_available_cols(
        top_dec,
        ["Label", "Diff_PctPts", "Lift_Pct", "P_Value", "Reliability_Band", "Clarity_Band", "Notes_Short"]
    )

st.divider()


# -----------------------------
# Overview visuals
# -----------------------------
st.subheader("Overview visuals")

fig_rel = executive_reliability_ribbon(view)
st.plotly_chart(fig_rel, use_container_width=True)

# Reliability reading help + takeaway
st.markdown(
    """
<div class="takeaway">
  <div class="takeaway-title">How to read “Reliability mix”</div>
  <div><b>What it shows:</b> how many KPI rows are highly trustworthy vs early signals vs low-confidence hints.</div>
  <div class="takeaway-sub"><b>Reliability</b> is driven by the smaller of the two samples (control vs exposed). If one side is small, the whole result is less stable.</div>
</div>
    """,
    unsafe_allow_html=True
)

if show_definitions:
    st.markdown(
        f"""
<div class="takeaway">
  <div class="takeaway-title">What each bar means</div>
  <div><b>Great:</b> very strong sample support (min sample ≥ {great_thr}). Safest results to cite.</div>
  <div><b>Good:</b> solid base (min sample ≥ {good_thr}). Usually reliable enough for decisions.</div>
  <div><b>Directional:</b> moderate base (min sample ≥ {dir_thr}). Useful as an early signal, not a guarantee.</div>
  <div><b>Low:</b> below {dir_thr}. Treat as a hint—avoid hard claims.</div>
</div>
        """,
        unsafe_allow_html=True
    )

st.divider()


# -----------------------------
# Comparison view
# -----------------------------
if show_comparison:
    st.subheader("Comparison view")

    st.markdown(
        """
<div class="takeaway">
  <div class="takeaway-title">What this section is for</div>
  <div>This section helps you prioritise the work: big changes vs noisy changes. You should act on the best combination of “impact” and “confidence”.</div>
  <div class="takeaway-sub">Use the matrix to see what to act on. Use the interval plot to see how stable the results are.</div>
</div>
        """,
        unsafe_allow_html=True
    )

    fig_matrix = executive_impact_matrix(view)
    st.plotly_chart(fig_matrix, use_container_width=True)

    st.markdown(
        """
<div class="takeaway">
  <div class="takeaway-title">How to read the Impact Matrix</div>
  <div><b>Right</b> means positive lift. <b>Left</b> means negative lift.</div>
  <div><b>Higher</b> means stronger evidence (lower p-value). <b>Lower</b> means weaker evidence.</div>
  <div class="takeaway-sub">Practical rule: top-right is where you get “worth acting on” results.</div>
</div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Effect sizes with confidence intervals (top 25)")
    st.plotly_chart(executive_forest_plot(view, top_n=min(25, len(view))), use_container_width=True)

    st.markdown(
        """
<div class="takeaway">
  <div class="takeaway-title">How to read the interval plot</div>
  <div>Each row is a KPI result. The dot is the gap (difference). The line is the plausible range for the true value.</div>
  <div><b>If the line crosses 0:</b> the result is not clearly confirmed.</div>
  <div class="takeaway-sub">This is the “evidence lens”: tight lines = stable results, wide lines = noisy results.</div>
</div>
        """,
        unsafe_allow_html=True
    )

st.divider()


# -----------------------------
# Results table
# -----------------------------
st.subheader("Results table (decision view)")

decision_cols = [
    "Month Year", "Brand", "Market", "Category", "KPI",
    "Control Sample", "Exposed Sample",
    "Control_Pct", "Exposed_Pct", "Diff_PctPts", "Lift_Pct",
    "Reliability_Band", "Clarity_Band", "Notes_Short"
]
show_df_with_available_cols(view, decision_cols)

st.divider()


# -----------------------------
# PDF Export
# -----------------------------
st.subheader("Export")

def make_card(r):
    # This card structure matches your pdf_report.py expectations:
    # card["state_label"], card["note"], card["meaning"], card["decision"]
    state = str(r.get("Clarity_Band", ""))
    rb = str(r.get("Reliability_Band", ""))
    gap = float(r.get("Diff_PctPts", 0.0))
    what = str(r.get("Human_WhatHappened", ""))
    conf = str(r.get("Human_Confidence", ""))

    state_label = "Clear result" if state == "Clear" else "Directional signal" if state == "Directional" else "Not confirmed"
    note = str(r.get("Notes_Short", "")).strip() or conf

    meaning = what
    if state != "Clear" or rb in ("Low", "Directional"):
        decision = "Treat as a signal to watch. If it matters, collect more responses or re-test."
    else:
        decision = "Safe to cite as evidence. Consider acting on this insight."

    return {
        "state_label": state_label,
        "note": note,
        "meaning": meaning,
        "decision": decision,
    }

if st.button("Generate PDF"):
    scope_df = view.copy() if export_scope == "All rows in view" else pd.DataFrame([row])
    cards = [make_card(r) for _, r in scope_df.iterrows()]
    pdf_bytes = build_pdf_bytes(scope_df, cards=cards, report_title=pdf_title, include_comparisons=True)
    st.download_button(
        "Download PDF",
        data=pdf_bytes,
        file_name=f"{pdf_title}.pdf",
        mime="application/pdf"
    )
