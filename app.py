import io
import pandas as pd
import streamlit as st

from src.io import read_uploaded_file, validate_input, take_only_inputs
from src.compute import compute_all_metrics
from src.charts import (
    executive_reliability_ribbon,
    executive_impact_matrix,
    executive_forest_plot,
    executive_story_card_chart,
)
from src.pdf_report import build_pdf_bytes
from src.story import (
    reliability_takeaway,
    impact_quadrant_takeaway,
    top_lists,
    ci_reading_help,
    matrix_reading_help,
)

st.set_page_config(page_title="BLS Brief", layout="wide")

CSS = """
<style>
/* Slightly tighter page + cleaner cards */
.block-container {padding-top: 2.0rem; padding-bottom: 2.0rem; max-width: 1250px;}
h1, h2, h3 {letter-spacing: -0.02em;}
.takeaway {
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 14px;
  padding: 14px 16px;
  background: rgba(255,255,255,0.65);
}
.takeaway-title {font-weight: 600; margin-bottom: 6px;}
.takeaway-sub {opacity: 0.75; font-size: 0.92rem; margin-top: 6px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# -----------------------------
# Header row: title + template button
# -----------------------------
c1, c2 = st.columns([0.78, 0.22], vertical_alignment="center")
with c1:
    st.title("BLS Brief")
    st.write("Upload inputs only. The platform calculates the stats (from scores + samples), shows interactive visuals, and exports a PDF.")
    st.caption("Template includes: Month Year, Brand, Category, Market, KPI, KPI Order (optional), Control Sample, Exposed Sample, Control Score, Exposed Score. No computed columns needed.")
with c2:
    # Download template (CSV)
    template_cols = [
        "Month Year","Brand","Category","Market","KPI","KPI Order",
        "Control Sample","Exposed Sample","Control Score","Exposed Score"
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
    strict_mode = st.checkbox("Strict input mode", value=False, help="If on, rejects files with missing required columns instead of trying to interpret them.")

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
uploaded = st.file_uploader("Upload CSV or XLSX", type=["csv","xlsx","xls"], label_visibility="collapsed")

if not uploaded:
    st.stop()

raw = read_uploaded_file(uploaded)

if strict_mode:
    validate_input(raw)

inputs = take_only_inputs(raw)

df = compute_all_metrics(inputs)

# Filters (simple)
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

if allow_removal:
    remove_non_def = st.checkbox("Remove non-definitive rows", value=False)
    if remove_non_def and "Reliability" in view.columns:
        view = view[~view["Reliability"].isin(["Directional", "Low"])]

if not include_non_def and "Reliability" in view.columns:
    view = view[~view["Reliability"].isin(["Directional", "Low"])]

st.divider()

# -----------------------------
# Summary
# -----------------------------
st.subheader("Summary")
s1, s2, s3, s4 = st.columns(4)

rows_in_view = len(view)
stat_clear = int((view.get("P_Value", 1) <= 0.05).sum())
avg_lift = float(view["Lift_Pct"].mean()) if "Lift_Pct" in view.columns and rows_in_view else 0.0
avg_gap = float(view["Diff_PctPts"].mean()) if "Diff_PctPts" in view.columns and rows_in_view else 0.0

s1.metric("Rows in view", f"{rows_in_view}")
s2.metric("Statistically clear", f"{stat_clear}")
s3.metric("Average lift", f"{avg_lift:.2f}%")
s4.metric("Average gap", f"{avg_gap:.2f} pts")

if show_definitions:
    st.markdown(
        """
<div class="takeaway">
  <div class="takeaway-title">What these numbers mean</div>
  <div><b>Rows in view:</b> how many KPI results you’re looking at after filters. Each row is one KPI result for a brand / market / month.</div>
  <div><b>Statistically clear:</b> how many rows have a strong enough signal that we can treat the change as real (95% confidence).</div>
  <div><b>Average lift:</b> typical relative change vs control across the rows you’re viewing. Useful for direction, not a single “overall performance score”.</div>
  <div><b>Average gap:</b> typical absolute difference between exposed and control, in percentage points. This is the “real-world” size of the change.</div>
</div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# -----------------------------
# Deep dive (moved up)
# -----------------------------
st.subheader("Deep dive")
if "Label" not in view.columns:
    view = view.copy()
    view["Label"] = view["Brand"].astype(str) + " • " + view["KPI"].astype(str) + " • " + view["Month Year"].astype(str)

choice = st.selectbox("Choose a row", view["Label"].tolist())
row = view[view["Label"] == choice].iloc[0]

# Headline box
kpi_name = str(row.get("KPI", ""))
brand_name = str(row.get("Brand", ""))
month = str(row.get("Month Year", ""))

gap = float(row.get("Diff_PctPts", 0))
lift = float(row.get("Lift_Pct", 0))
pval = float(row.get("P_Value", 1))
rel = str(row.get("Reliability", ""))

direction = "increase" if gap > 0 else "decline" if gap < 0 else "no change"
tag = "clear result" if pval <= 0.05 else "directional result"

st.markdown(
    f"""
<div class="takeaway">
  <div class="takeaway-title">{brand_name} • {kpi_name}</div>
  <div style="opacity:0.75;">{month} • {str(row.get("Category",""))} • {str(row.get("Market",""))} • <b>{tag}</b></div>
  <br/>
  <div><b>What changed:</b> {kpi_name} shows a {direction} of <b>{gap:.2f} points</b> in exposed vs control.</div>
  <div><b>Is it real:</b> p={pval:.4f} → {"unlikely to be random noise" if pval<=0.05 else "could be noise; treat as a signal to watch"}.</div>
  <div><b>How to use it:</b> {"Safe to cite as evidence and act on." if pval<=0.05 else "Don’t make a hard claim yet—monitor or test again."}</div>
</div>
    """,
    unsafe_allow_html=True,
)

# Mini chart
st.plotly_chart(executive_story_card_chart(row), use_container_width=True)

st.divider()

# -----------------------------
# What moved most / least
# -----------------------------
st.subheader("What moved most / least (quick scan)")
left, right = st.columns(2)

top_inc = view.sort_values("Diff_PctPts", ascending=False).head(7)
top_dec = view.sort_values("Diff_PctPts", ascending=True).head(7)

with left:
    st.caption("Top increases")
    st.dataframe(top_inc[["Label","Diff_PctPts","Lift_Pct","P_Value","Reliability"]], use_container_width=True)
with right:
    st.caption("Top declines")
    st.dataframe(top_dec[["Label","Diff_PctPts","Lift_Pct","P_Value","Reliability"]], use_container_width=True)

st.divider()

# -----------------------------
# Overview visuals
# -----------------------------
st.subheader("Overview visuals")

fig_rel = executive_reliability_ribbon(view)
st.plotly_chart(fig_rel, use_container_width=True)

take = reliability_takeaway(view)
st.markdown(
    f"""
<div class="takeaway">
  <div class="takeaway-title">{take["headline"]}</div>
  <div>{take["message"]}</div>
  <div class="takeaway-sub">{take["sub"]}</div>
</div>
    """,
    unsafe_allow_html=True
)

if show_definitions:
    st.markdown(
        """
<div class="takeaway">
  <div class="takeaway-title">How to read the reliability chart</div>
  <div><b>High:</b> clear result + healthy sample sizes. Safe to use as a conclusion.</div>
  <div><b>Medium:</b> clear result, but sample is not ideal. Still useful, but be cautious.</div>
  <div><b>Directional:</b> not statistically clear, but shows a pattern worth watching. Treat as a signal, not a conclusion.</div>
  <div><b>Low:</b> too little data (or too noisy). Avoid using for decisions.</div>
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
  <div>This view helps you prioritise. It separates “big changes” from “noisy changes” so you don’t chase randomness.</div>
  <div style="opacity:0.75; margin-top:6px;">Use the impact matrix first (action lens), then the confidence-interval ranking (evidence lens).</div>
</div>
        """,
        unsafe_allow_html=True
    )

    fig_matrix = executive_impact_matrix(view)
    st.plotly_chart(fig_matrix, use_container_width=True)

    tq = impact_quadrant_takeaway(view)
    st.markdown(
        f"""
<div class="takeaway">
  <div class="takeaway-title">{tq["headline"]}</div>
  <div>{tq["message"]}</div>
  <div class="takeaway-sub">{tq["sub"]}</div>
</div>
        """,
        unsafe_allow_html=True
    )

    opp, risk = top_lists(view, n=5)
    cA, cB = st.columns(2)
    with cA:
        st.caption("Top opportunities (Act)")
        st.dataframe(opp, use_container_width=True)
    with cB:
        st.caption("Top risks (Investigate)")
        st.dataframe(risk, use_container_width=True)

    if show_definitions:
        helpbox = matrix_reading_help()
        st.markdown(
            f"""
<div class="takeaway">
  <div class="takeaway-title">{helpbox["title"]}</div>
  <pre style="white-space:pre-wrap; margin:0; font-family: inherit;">{helpbox["text"]}</pre>
  <div class="takeaway-sub">How it’s calculated: lift comes from exposed vs control scores. Certainty comes from the p-value of the difference test.</div>
</div>
            """,
            unsafe_allow_html=True
        )

    st.subheader("Effect sizes with confidence intervals (top 25)")
    st.plotly_chart(executive_forest_plot(view, top_n=min(25, len(view))), use_container_width=True)

    if show_definitions:
        ci = ci_reading_help()
        st.markdown(
            f"""
<div class="takeaway">
  <div class="takeaway-title">{ci["title"]}</div>
  <pre style="white-space:pre-wrap; margin:0; font-family: inherit;">{ci["text"]}</pre>
</div>
            """,
            unsafe_allow_html=True
        )

st.divider()

# -----------------------------
# Results table + Export
# -----------------------------
st.subheader("Results table")
st.dataframe(view, use_container_width=True)

st.subheader("Export")
if st.button("Generate PDF"):
    scope_df = view.copy() if export_scope == "All rows in view" else pd.DataFrame([row])
    pdf_bytes = build_pdf_bytes(scope_df, title=pdf_title)
    st.download_button("Download PDF", data=pdf_bytes, file_name=f"{pdf_title}.pdf", mime="application/pdf")
