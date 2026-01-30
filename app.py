import io
import json
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
.block-container {padding-top: 1.8rem; padding-bottom: 2.0rem; max-width: 1280px;}
h1, h2, h3 {letter-spacing: -0.02em;}
.takeaway {
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 14px;
  padding: 14px 16px;
  background: rgba(255,255,255,0.65);
}
.takeaway-title {font-weight: 650; margin-bottom: 6px;}
.takeaway-sub {opacity: 0.75; font-size: 0.92rem; margin-top: 6px;}
.kpi-pill {
  display:inline-block; padding: 3px 10px; border-radius: 999px;
  border: 1px solid rgba(0,0,0,0.10); font-size: 0.85rem; opacity: 0.9;
  margin-right: 6px;
}
.note {
  border: 1px dashed rgba(0,0,0,0.15);
  border-radius: 12px;
  padding: 10px 12px;
  background: rgba(255,255,255,0.55);
  font-size: 0.95rem;
  line-height: 1.35rem;
}
.small {opacity: 0.72; font-size: 0.92rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def _badge(label: str):
    return f'<span class="kpi-pill">{label}</span>'


def _safe_float(x, default=0.0):
    try:
        v = float(x)
        return v
    except Exception:
        return default


def _notes_list(row):
    try:
        keys = json.loads(row.get("Notes_Keys", "[]"))
        return keys if isinstance(keys, list) else []
    except Exception:
        return []


# -----------------------------
# Header row + template
# -----------------------------
c1, c2 = st.columns([0.75, 0.25], vertical_alignment="center")
with c1:
    st.title("BLS Brief")
    st.write("Upload inputs only. The platform calculates the stats, explains them in plain language, shows interactive visuals, and exports aI-safe PDFs.")
    with st.expander("How to read this (60 seconds)", expanded=False):
        st.markdown(
            """
- **Each row** is one KPI result for a brand / market / time period.
- **Gap (pp)** is the real-world difference: “out of 100 people, how many more said yes”.
- **Lift (%)** is the relative change. It can be misleading
