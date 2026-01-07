import streamlit as st
import pandas as pd

from src.io import read_uploaded_file, validate_input, make_brand_alias_map, apply_aliases
from src.metrics import compute_metrics
from src.insights import build_insight_cards
from src.pdf_report import build_pdf_bytes

# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
st.set_page_config(
    page_title="BLS Brief",
    layout="wide"
)

st.title("BLS Brief")
st.caption("Upload brand lift data. Review results clearly. Export a clean PDF.")

# ------------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------------
with st.sidebar:
    st.subheader("Analysis options")

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
        "Enable comparison views",
        value=True
    )

    st.divider()

    st.subheader("Report")
    report_title = st.text_input(
        "PDF report title",
        value="BLS Brief"
    )

# ------------------------------------------------------------------
# File upload
# ------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload CSV or XLSX file",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is None:
    st.info("Upload a file to begin. Brand names will be automatically aliased.")
    st.stop()

# ------------------------------------------------------------------
# Read + validate input
# ------------------------------------------------------------------
try:
    raw_df = read_uploaded_file(uploaded_file)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

validation = validate_input(raw_df)

if not validation["ok"]:
    st.error("The uploaded file does not match the required structure.")
    st.write("Missing required columns:", validation["missing_base"])
    st.write(
        "The file must contain either:\n"
        "- Control Score + Exposed Score, or\n"
        "- Control_Prop + Exposed_Prop"
    )
    st.write("Columns detected:", validation["columns"])
    st.stop()

# ------------------------------------------------------------------
# Alias brands
# ------------------------------------------------------------------
alias_map = make_brand_alias_map(raw_df["Brand"].tolist())
df = apply_aliases(raw_df, alias_map)

# ------------------------------------------------------------------
# Compute metrics
# ------------------------------------------------------------------
df_metrics = compute_metrics(df)

# ------------------------------------------------------------------
# Filters
# ------------------------------------------------------------------
st.subheader("Filters")

c1, c2, c3, c4 = st.columns(4)

with c1:
    market_filter = st.multiselect(
        "Market",
        sorted(df_metrics["Market"].dropna().astype(str).unique().tolist())
    )

with c2:
    category_filter = st.multiselect(
        "Category",
        sorted(df_metrics["Category"].dropna().astype(str).unique().tolist())
    )

with c3:
    brand_filter = st.multiselect(
        "Brand",
        sorted(df_metrics["Brand_Alias"].dropna().astype(str).unique().tolist())
    )

with c4:
    kpi_filter = st.multiselect(
        "KPI",
        sorted(df_metrics["KPI"].dropna().astype(str).unique().tolist())
    )

filtered_df = df_metrics.copy()

if market_filter:
    filtered_df = filtered_df[filtered_df["Market"].astype(str).isin(market_filter)]

if category_filter:
    filtered_df = filtered_df[filtered_df["Category"].astype(str).isin(category_filter)]

if brand_filter:
    filtered_df = filtered_df[filtered_df["Brand_Alias"].astype(str).isin(brand_filter)]

if kpi_filter:
    filtered_df = filtered_df[filtered_df["KPI"].astype(str).isin(kpi_filter)]

# ------------------------------------------------------------------
# Optional exclusion of non-definitive rows
# ------------------------------------------------------------------
if allow_exclude_non_sig:
    remove_non_sig = st.checkbox(
        "Remove non-definitive rows from analysis",
        value=False
    )
else:
    remove_non_sig = False

if remove_non_sig:
    filtered_df = filtered_df[filtered_df["Significant_95"] == True]
    include_non_sig_effective = False
else:
    include_non_sig_effective = include_non_sig

# ------------------------------------------------------------------
# Results table
# ------------------------------------------------------------------
st.subheader("Results")

display_columns = [
    "Month Year",
    "Brand_Alias",
    "Category",
    "Market",
    "KPI",
    "Control Sample",
    "Exposed Sample",
    "Control_Pct",
    "Exposed_Pct",
    "Diff_PctPts",
    "Lift_Pct",
    "P_Value",
    "Significant_95",
    "Data_Flag"
]

existing_cols = [c for c in display_columns if c in filtered_df.columns]

st.dataframe(
    filtered_df[existing_cols].reset_index(drop=True),
    use_container_width=True,
    height=360
)

# ------------------------------------------------------------------
# Insight notes
# ------------------------------------------------------------------
st.subheader("Interpretation")

if len(filtered_df) == 0:
    st.warning("No rows available with the current filters.")
    st.stop()

insight_cards = build_insight_cards(
    filtered_df.reset_index(drop=True),
    include_non_sig=include_non_sig_effective
)

for i, row in enumerate(filtered_df.reset_index(drop=True).itertuples(index=False)):
    card = insight_cards[i]

    with st.container(border=True):
        left, right = st.columns([2.5, 1])

        with left:
            st.markdown(
                f"**{row.Brand_Alias} • {row.KPI} • {getattr(row, 'Month Year', '')}**"
            )
            st.write(f"{card['state_label']} — {card['note']}")
            st.write(card["meaning"])
            st.write(card["decision"])

        with right:
            st.metric("Lift (%)", f"{row.Lift_Pct:.2f}")
            st.metric("Diff (pts)", f"{row.Diff_PctPts:.2f}")
            st.caption(f"p-value: {row.P_Value:.4f}")

# ------------------------------------------------------------------
# Comparison view (optional)
# ------------------------------------------------------------------
if compare_mode and len(filtered_df) > 1:
    st.subheader("Comparison view")
    st.caption("Optional view to understand relative movement across rows.")

    temp = filtered_df.copy()
    temp["Label"] = temp["Brand_Alias"].astype(str) + " • " + temp["KPI"].astype(str)

    st.bar_chart(
        temp.set_index("Label")["Lift_Pct"],
        use_container_width=True
    )

# ------------------------------------------------------------------
# PDF export
# ------------------------------------------------------------------
st.divider()
st.subheader("Export")

if st.button("Generate PDF"):
    try:
        pdf_bytes = build_pdf_bytes(
            filtered_df.reset_index(drop=True),
            insight_cards,
            report_title=report_title,
            include_comparisons=compare_mode
        )

        st.download_button(
            label="Download PDF",
            data=pdf_bytes,
            file_name="BLS_brief.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.error(f"PDF generation failed: {e}")

# ------------------------------------------------------------------
# Alias mapping (internal visibility only)
# ------------------------------------------------------------------
with st.expander("Brand alias mapping (internal)"):
    st.dataframe(
        pd.DataFrame({
            "Original brand": list(alias_map.keys()),
            "Alias used": list(alias_map.values())
        }),
        use_container_width=True
    )
