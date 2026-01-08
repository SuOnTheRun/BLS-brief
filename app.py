import streamlit as st
import pandas as pd

from src.io import read_uploaded_file, validate_input, take_only_inputs
from src.metrics import compute_metrics
from src.insights import build_insight_cards
from src.pdf_report import build_pdf_bytes
from src.charts import interactive_dumbbell, interactive_lift_rank, interactive_confidence_scatter

CSS = """
<style>
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }
h1, h2, h3 { letter-spacing: -0.02em; }
.small-muted { color: rgba(49, 51, 63, 0.65); font-size: 0.92rem; }
.card {
  border: 1px solid rgba(49, 51, 63, 0.12);
  border-radius: 14px;
  padding: 14px 16px;
  background: white;
}
.card-title { font-size: 0.85rem; color: rgba(49, 51, 63, 0.65); margin-bottom: 6px; }
.card-value { font-size: 1.25rem; font-weight: 750; margin: 0; }
.pill {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid rgba(49, 51, 63, 0.15);
  font-size: 0.8rem;
  color: rgba(49, 51, 63, 0.75);
  background: rgba(49, 51, 63, 0.03);
}
.hr { height: 1px; background: rgba(49, 51, 63, 0.10); margin: 14px 0; }
.kpi-row { border: 1px solid rgba(49,51,63,0.10); border-radius: 14px; padding: 14px 16px; background: white; }
.kpi-head { font-weight: 750; font-size: 1.0rem; margin-bottom: 6px; }
.kpi-sub { color: rgba(49,51,63,0.65); font-size: 0.85rem; margin-bottom: 10px; }
.note { color: rgba(49,51,63,0.85); font-size: 0.95rem; line-height: 1.45; }
</style>
"""

st.set_page_config(page_title="BLS Brief", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

st.markdown("## BLS Brief")
st.markdown(
    '<div class="small-muted">Upload inputs only. The platform computes all statistical columns and exports a clean PDF.</div>',
    unsafe_allow_html=True
)

with st.sidebar:
    st.subheader("Settings")

    strict_mode = st.checkbox(
        "Strict input mode",
        value=False,
        help="If on: the upload is rejected when extra (computed) columns are present. If off: extra columns are ignored."
    )

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
        value=True
    )

    st.divider()
    st.subheader("PDF")
    report_title = st.text_input("Title", value="BLS Brief")
    pdf_scope = st.radio("Export scope", ["Selected row only", "All rows in view"], index=0)

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

# Validate input surface
check = validate_input(raw)

if not check["ok"]:
    st.error("Your upload is missing required input columns.")
    st.write("Missing base columns:", check["missing_base"])
    st.write("Missing score columns:", check["missing_scores"])
    st.write("Allowed inputs:", check["allowed_inputs"])
    st.stop()

# Extras warning or strict block
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

# Enforce: keep only input surface (drops extra columns)
inputs = take_only_inputs(raw)

# Compute everything from inputs
df = compute_metrics(inputs)

# Filters
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
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

# Summary
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
st.markdown("### Summary")

total = len(filtered)
clear_count = int(filtered["Significant_95"].sum())
avg_lift = float(filtered["Lift_Pct"].mean())

cc1, cc2, cc3, cc4 = st.columns(4)
with cc1:
    st.markdown(f'<div class="card"><div class="card-title">Rows in view</div><div class="card-value">{total}</div></div>', unsafe_allow_html=True)
with cc2:
    st.markdown(f'<div class="card"><div class="card-title">Statistically clear</div><div class="card-value">{clear_count}</div></div>', unsafe_allow_html=True)
with cc3:
    st.markdown(f'<div class="card"><div class="card-title">Average lift</div><div class="card-value">{avg_lift:.2f}%</div></div>', unsafe_allow_html=True)
with cc4:
    mode = "Non-definitive included" if include_non_sig_effective else "Non-definitive removed"
    st.markdown(f'<div class="card"><div class="card-title">Mode</div><div class="card-value">{mode}</div></div>', unsafe_allow_html=True)

# Results table
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
st.markdown("### Results table")

table_cols = [
    "Month Year", "Brand", "Category", "Market", "KPI",
    "Control Sample", "Exposed Sample",
    "Control_Pct", "Exposed_Pct",
    "Diff_PctPts", "Lift_Pct",
    "Z_Score", "P_Value", "Significant_95", "Data_Flag"
]
existing = [c for c in table_cols if c in filtered.columns]
st.dataframe(filtered[existing].reset_index(drop=True), use_container_width=True, height=320)

# Deep dive (one row at a time)
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
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
pill = card["state_label"]

st.markdown(
    f"""
    <div class="kpi-row">
      <div class="kpi-head">{row.get('Brand','')} • {row.get('KPI','')}</div>
      <div class="kpi-sub">{meta} &nbsp; <span class="pill">{pill}</span></div>
      <div class="note"><b>Note:</b> {card["note"]}</div>
      <div class="note"><b>What changed:</b> {card["meaning"]}</div>
      <div class="note"><b>How to use it:</b> {card["decision"]}</div>
    </div>
    """,
    unsafe_allow_html=True
)

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(f'<div class="card"><div class="card-title">Lift</div><div class="card-value">{row["Lift_Pct"]:.2f}%</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="card"><div class="card-title">Difference</div><div class="card-value">{row["Diff_PctPts"]:.2f} pts</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="card"><div class="card-title">p-value</div><div class="card-value">{row["P_Value"]:.4f}</div></div>', unsafe_allow_html=True)

st.plotly_chart(interactive_dumbbell(row), use_container_width=True)

# Comparison (interactive)
if compare_mode and len(filtered) > 1:
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown("### Comparison view")
    st.plotly_chart(interactive_lift_rank(filtered), use_container_width=True)
    st.plotly_chart(interactive_confidence_scatter(filtered), use_container_width=True)

# Export
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
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
