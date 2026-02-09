import io
import pandas as pd
import streamlit as st

from src.io import read_uploaded_file, validate_input, take_only_inputs
from src.metrics import compute_all_metrics
from src.charts import (
    executive_reliability_ribbon,
    executive_clarity_mix,
    executive_impact_matrix,
    executive_forest_plot,
    executive_story_card_chart,
)
from src.insights import (
    add_insights_columns,
    write_view_insights,
    write_row_insight,
)

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="BLS Brief", layout="wide")

CSS = """
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1250px;}
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

# --------------------------------------------------
# Header + template download
# --------------------------------------------------
c1, c2 = st.columns([0.78, 0.22], vertical_alignment="center")

with c1:
    st.title("BLS Brief")
    st.write(
        "Upload **inputs only**. The platform calculates the statistics, "
        "explains what moved, how confident we are, and what to do next."
    )
    st.caption(
        "Template columns only — no calculated fields required."
    )

with c2:
    template_cols = [
        "Month Year",
        "Market",
        "Brand",
        "Category",
        "KPI",
        "Control Sample",
        "Exposed Sample",
        "Control Score",
        "Exposed Score",
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

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.header("Input handling")
    strict_mode = st.checkbox(
        "Strict input mode",
        value=False,
        help="Reject files with missing required columns instead of trying to interpret them."
    )

    st.header("View options")
    include_non_def = st.checkbox("Include non-definitive results", value=True)
    allow_removal = st.checkbox("Allow removal of non-definitive rows", value=True)
    show_definitions = st.checkbox("Show definitions & explanations", value=True)

# --------------------------------------------------
# Upload
# --------------------------------------------------
st.subheader("Upload CSV or Excel")
uploaded = st.file_uploader(
    "Upload CSV or XLSX",
    type=["csv", "xlsx", "xls"],
    label_visibility="collapsed"
)

if not uploaded:
    st.stop()

raw = read_uploaded_file(uploaded)

if strict_mode:
    validate_input(raw)

inputs = take_only_inputs(raw)

# --------------------------------------------------
# Compute metrics + insights
# --------------------------------------------------
df = compute_all_metrics(inputs)
df = add_insights_columns(df)

# --------------------------------------------------
# Filters
# --------------------------------------------------
st.subheader("Filters")

f1, f2, f3, f4 = st.columns(4)

def pick(col, container):
    if col not in df.columns:
        return None
    opts = sorted([x for x in df[col].dropna().unique()])
    return container.selectbox(col, ["(All)"] + opts)

market = pick("Market", f1)
brand = pick("Brand", f2)
category = pick("Category", f3)
kpi = pick("KPI", f4)

view = df.copy()
for col, val in [
    ("Market", market),
    ("Brand", brand),
    ("Category", category),
    ("KPI", kpi),
]:
    if val and val != "(All)" and col in view.columns:
        view = view[view[col] == val]

if allow_removal:
    remove_non_def = st.checkbox("Remove non-definitive rows", value=False)
    if remove_non_def:
        view = view[~view["Reliability_Band"].isin(["Directional", "Low"])]

if not include_non_def:
    view = view[~view["Reliability_Band"].isin(["Directional", "Low"])]

st.divider()

# --------------------------------------------------
# Summary
# --------------------------------------------------
st.subheader("Summary")

rows_in_view = len(view)
stat_clear = int((view["Clarity_Band"] == "Clear").sum()) if rows_in_view else 0
avg_gap = float(view["Diff_PctPts"].mean()) if rows_in_view else 0.0

cA, cB, cC = st.columns(3)
cA.metric("Rows in view", rows_in_view)
cB.metric("Statistically clear", stat_clear)
cC.metric("Average gap", f"{avg_gap:.2f} pts")

if show_definitions:
    st.markdown(
        """
<div class="takeaway">
<b>Rows in view</b>: how many KPI results you’re looking at after filters.<br/>
<b>Statistically clear</b>: results where the signal is strong enough to treat as real.<br/>
<b>Average gap</b>: typical difference between exposed and control, in percentage points.
</div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# --------------------------------------------------
# Auto-written insights (VIEW LEVEL)
# --------------------------------------------------
st.subheader("Insights")

view_pack = write_view_insights(view, top_n=5)

st.markdown(
    f"""
<div class="takeaway">
  <div class="takeaway-title">{view_pack["headline"]}</div>

  <div style="margin-top:10px;"><b>Top wins</b></div>
  <ul>{"".join([f"<li>{x}</li>" for x in view_pack["wins"]]) if view_pack["wins"] else "<li>No clear wins yet.</li>"}</ul>

  <div><b>Top risks</b></div>
  <ul>{"".join([f"<li>{x}</li>" for x in view_pack["risks"]]) if view_pack["risks"] else "<li>No clear risks yet.</li>"}</ul>

  <div><b>Watch list</b></div>
  <ul>{"".join([f"<li>{x}</li>" for x in view_pack["watch"]]) if view_pack["watch"] else "<li>No directional signals needing confirmation.</li>"}</ul>
</div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# --------------------------------------------------
# Deep dive
# --------------------------------------------------
st.subheader("Deep dive")

if "Label" not in view.columns:
    view["Label"] = (
        view["Brand"].astype(str)
        + " • "
        + view["KPI"].astype(str)
        + " • "
        + view["Month Year"].astype(str)
    )

choice = st.selectbox("Choose a result", view["Label"].tolist())
row = view[view["Label"] == choice].iloc[0]

ins = write_row_insight(row)

st.markdown(
    f"""
<div class="takeaway">
  <div class="takeaway-title">Insight</div>
  <div><b>{ins["headline"]}</b></div>
  <div style="margin-top:6px;">{ins["evidence"]}</div>
  <div style="margin-top:6px;"><b>So what:</b> {ins["so_what"]}</div>
  {"<div style='margin-top:8px; opacity:0.85;'><b>Notes:</b><br/>" + "<br/>".join(ins["notes"]) + "</div>" if ins["notes"] else ""}
</div>
    """,
    unsafe_allow_html=True,
)

st.plotly_chart(
    executive_story_card_chart(row),
    use_container_width=True
)

st.divider()

# --------------------------------------------------
# Overview visuals
# --------------------------------------------------
st.subheader("Overview visuals")

st.plotly_chart(
    executive_reliability_ribbon(view),
    use_container_width=True
)

st.plotly_chart(
    executive_clarity_mix(view),
    use_container_width=True
)

st.divider()

# --------------------------------------------------
# Comparison views
# --------------------------------------------------
st.subheader("Comparison")

st.plotly_chart(
    executive_impact_matrix(view),
    use_container_width=True
)

st.plotly_chart(
    executive_forest_plot(view, top_n=min(25, len(view))),
    use_container_width=True
)

st.divider()

# --------------------------------------------------
# Results table
# --------------------------------------------------
st.subheader("Results table")

cols = [
    "Month Year",
    "Market",
    "Brand",
    "Category",
    "KPI",
    "Control Sample",
    "Exposed Sample",
    "Control_Pct",
    "Exposed_Pct",
    "Diff_PctPts",
    "Lift_Pct",
    "P_Value",
    "Reliability_Band",
    "Clarity_Band",
    "Insight_Headline",
]

cols = [c for c in cols if c in view.columns]
st.dataframe(view[cols], use_container_width=True)
