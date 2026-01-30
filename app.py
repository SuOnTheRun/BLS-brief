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
- **Lift (%)** is the relative change. It can be misleading when the baseline is tiny, so the platform hides it when needed.
- **Reliability** is mostly sample size: more answers = more stability.
- **Clarity** is whether the difference looks real vs random variation.
"""
        )

with c2:
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
    strict_mode = st.checkbox("Strict input mode", value=False, help="Rejects files with missing required columns.")

    st.header("Reliability bands (movable)")
    great_thr = st.slider("Great (min sample ≥)", 150, 600, 300, 10)
    good_thr = st.slider("Good (min sample ≥)", 50, 300, 100, 5)
    directional_thr = st.slider("Directional (min sample ≥)", 10, 150, 50, 5)

    st.header("Clarity thresholds")
    clear_p = st.slider("Clear p-threshold", 0.01, 0.10, 0.05, 0.01)
    directional_p = st.slider("Directional p-threshold", 0.05, 0.25, 0.10, 0.01)

    st.header("Lift rules")
    min_baseline = st.slider("Hide lift if baseline < (%)", 0, 20, 5, 1)
    min_lift_sample = st.slider("Hide lift if Control n < ", 10, 200, 50, 5)

    st.header("Views")
    remove_unclear = st.checkbox("Remove unclear results", value=False, help="Drops rows marked as Unclear or Low reliability.")
    show_comparison = st.checkbox("Comparison view", value=True)
    show_defs = st.checkbox("Show plain-English definitions", value=True)

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

df = compute_all_metrics(
    inputs,
    GREAT_THRESHOLD=great_thr,
    GOOD_THRESHOLD=good_thr,
    DIRECTIONAL_THRESHOLD=directional_thr,
    CLEAR_P_THRESHOLD=clear_p,
    DIRECTIONAL_P_THRESHOLD=directional_p,
    MIN_BASELINE_PERCENT=float(min_baseline),
    MIN_LIFT_SAMPLE=int(min_lift_sample),
)

# -----------------------------
# Filters (order of human thinking)
# -----------------------------
st.subheader("Filters")
f1, f2, f3, f4 = st.columns(4)

def pick(col, container):
    if col not in df.columns:
        return None
    opts = sorted([x for x in df[col].dropna().unique()])
    return container.selectbox(col, ["(All)"] + opts, key=f"f_{col}")

market = pick("Market", f1)
brand = pick("Brand", f2)
month = pick("Month Year", f3)
kpi = pick("KPI", f4)

view = df.copy()
for col, val in [("Market", market), ("Brand", brand), ("Month Year", month), ("KPI", kpi)]:
    if val and val != "(All)" and col in view.columns:
        view = view[view[col] == val]

# Optional removal rule
if remove_unclear:
    view = view[~view["Clarity_Band"].isin(["Unclear"])]
    view = view[~view["Reliability_Band"].isin(["Low"])]

# Filter feedback sentence
def _pick_label(val):
    return val if val and val != "(All)" else "All"

st.caption(
    f"Showing **{_pick_label(brand)}** ({_pick_label(market)}), **{_pick_label(month)}**, KPI: **{_pick_label(kpi)}** — **{len(view)}** result(s)."
)

st.divider()


# -----------------------------
# Executive summary (4 cards)
# -----------------------------
st.subheader("Summary (headline)")
s1, s2, s3, s4 = st.columns(4)

if len(view) == 0:
    st.info("No rows match the current filters.")
    st.stop()

tmp = view.copy()
tmp["AbsGap"] = tmp["Diff_PctPts"].astype(float).abs()

# Biggest win / risk
biggest_win = tmp.sort_values("Diff_PctPts", ascending=False).iloc[0]
biggest_risk = tmp.sort_values("Diff_PctPts", ascending=True).iloc[0]

# Most reliable (prefer Great/Good + Clear, then highest Reliability_N)
rank_rel = tmp.copy()
rank_rel["RelRank"] = rank_rel["Reliability_Band"].map({"Great": 4, "Good": 3, "Directional": 2, "Low": 1}).fillna(1)
rank_rel["ClaRank"] = rank_rel["Clarity_Band"].map({"Clear": 3, "Directional": 2, "Unclear": 1}).fillna(1)
rank_rel = rank_rel.sort_values(["ClaRank","RelRank","Reliability_N","AbsGap"], ascending=False)
most_reliable = rank_rel.iloc[0]

# Needs caution (lowest reliability or unclear)
rank_caution = tmp.copy()
rank_caution["RelRank"] = rank_caution["Reliability_Band"].map({"Great": 4, "Good": 3, "Directional": 2, "Low": 1}).fillna(1)
rank_caution["ClaRank"] = rank_caution["Clarity_Band"].map({"Clear": 3, "Directional": 2, "Unclear": 1}).fillna(1)
rank_caution = rank_caution.sort_values(["ClaRank","RelRank","Reliability_N"], ascending=True)
needs_caution = rank_caution.iloc[0]

def _card_sentence(r):
    k = str(r.get("KPI",""))
    gap = _safe_float(r.get("Diff_PctPts", 0.0))
    direction = "more" if gap > 0 else "fewer" if gap < 0 else "about the same"
    return f"Out of 100 people, **{abs(gap):.1f} {direction}** said yes after seeing the ads."

def _nums_line(r):
    return f"Control {r.get('Control_Pct',0):.1f}% (n={int(r.get('Control Sample',0))}) • Exposed {r.get('Exposed_Pct',0):.1f}% (n={int(r.get('Exposed Sample',0))})"

def _meta_line(r):
    return f"{r.get('Brand','')} • {r.get('KPI','')} • {r.get('Month Year','')}"

with s1:
    st.metric("Biggest Win", f"{_safe_float(biggest_win.get('Diff_PctPts')):.1f} pp")
    st.caption(_meta_line(biggest_win))
    st.write(_card_sentence(biggest_win))
    st.caption(_nums_line(biggest_win))

with s2:
    st.metric("Biggest Risk", f"{_safe_float(biggest_risk.get('Diff_PctPts')):.1f} pp")
    st.caption(_meta_line(biggest_risk))
    st.write(_card_sentence(biggest_risk))
    st.caption(_nums_line(biggest_risk))

with s3:
    st.metric("Most Reliable", f"{most_reliable.get('Reliability_Band')} / {most_reliable.get('Clarity_Band')}")
    st.caption(_meta_line(most_reliable))
    st.write(_card_sentence(most_reliable))
    st.caption(f"min n={int(most_reliable.get('Reliability_N',0))} • p={_safe_float(most_reliable.get('P_Value',1)):.4f}")

with s4:
    st.metric("Needs Caution", f"{needs_caution.get('Reliability_Band')} / {needs_caution.get('Clarity_Band')}")
    st.caption(_meta_line(needs_caution))
    st.write(_card_sentence(needs_caution))
    st.caption(needs_caution.get("Notes_Short",""))

if show_defs:
    st.markdown(
        """
<div class="takeaway">
  <div class="takeaway-title">Definitions (plain English)</div>
  <div><b>Gap (pp):</b> the real difference. “Out of 100 people, how many more said yes?”</div>
  <div><b>Lift (%):</b> the relative change vs control. Useful, but can exaggerate tiny baselines — so we hide it when needed.</div>
  <div><b>Reliability:</b> mostly sample size. More answers = more stable result.</div>
  <div><b>Clarity:</b> whether the difference looks real vs random variation (based on p-value + reliability gate).</div>
</div>
        """,
        unsafe_allow_html=True,
    )

st.divider()


# -----------------------------
# Deep dive (up the page)
# -----------------------------
st.subheader("Deep dive (one KPI result)")

view = view.copy()
if "Label" not in view.columns:
    view["Label"] = (
        view.get("Brand","").astype(str)
        + " • " + view.get("KPI","").astype(str)
        + " • " + view.get("Month Year","").astype(str)
    )

choice = st.selectbox("Choose a row", view["Label"].tolist())
row = view[view["Label"] == choice].iloc[0]

gap = _safe_float(row.get("Diff_PctPts", 0.0))
lift = row.get("Lift_Pct", float("nan"))
lift_visible = bool(row.get("Lift_Visible", True))

rel_band = str(row.get("Reliability_Band",""))
cla_band = str(row.get("Clarity_Band",""))
pval = _safe_float(row.get("P_Value", 1.0))

direction = "increase" if gap > 0 else "decline" if gap < 0 else "no change"

st.markdown(
    f"""
<div class="takeaway">
  <div class="takeaway-title">{row.get("Brand","")} • {row.get("KPI","")}</div>
  <div class="small">{row.get("Month Year","")} • {row.get("Market","")} • {row.get("Category","")}</div>
  <br/>
  <div>{_badge("What changed")} {direction} of <b>{gap:.2f} points</b> (exposed minus control).</div>
  <div>{_badge("How sure")} Reliability: <b>{rel_band}</b> (min n={int(row.get("Reliability_N",0))}) • Clarity: <b>{cla_band}</b> • p={pval:.4f}</div>
</div>
    """,
    unsafe_allow_html=True,
)

# Story chart
st.plotly_chart(executive_story_card_chart(row), use_container_width=True)

# Plain-English interpretation
control_pct = _safe_float(row.get("Control_Pct", 0.0))
exposed_pct = _safe_float(row.get("Exposed_Pct", 0.0))
st.markdown(
    f"""
<div class="takeaway">
  <div class="takeaway-title">What this means (in human terms)</div>
  <div><b>Out of 100 people:</b> about <b>{exposed_pct:.1f}</b> said “yes” after seeing the ad vs <b>{control_pct:.1f}</b> who didn’t.</div>
  <div><b>Difference:</b> that’s <b>{gap:.1f} more/fewer</b> out of 100 people.</div>
  <div><b>Recommendation strength:</b> {"safe to cite and act on" if cla_band=="Clear" and rel_band in ["Good","Great"] else "treat as a signal (needs caution)"}.</div>
</div>
    """,
    unsafe_allow_html=True,
)

# Notes (if any)
if row.get("Notes_Full",""):
    st.markdown(f'<div class="note"><b>Notes</b><br/>{row.get("Notes_Full","").replace(chr(10), "<br/>")}</div>', unsafe_allow_html=True)

st.divider()


# -----------------------------
# Overview visuals
# -----------------------------
st.subheader("Overview visuals")

st.plotly_chart(executive_reliability_ribbon(view), use_container_width=True)

st.markdown(
    """
<div class="takeaway">
  <div class="takeaway-title">How to read “Reliability mix”</div>
  <div><b>Great / Good:</b> enough people answered. More stable, safer to use.</div>
  <div><b>Directional:</b> moderate sample. Treat as an early signal.</div>
  <div><b>Low:</b> small sample. Don’t use as a hard conclusion.</div>
</div>
    """,
    unsafe_allow_html=True
)

st.divider()


# -----------------------------
# Comparison view
# -----------------------------
if show_comparison:
    st.subheader("Comparison view (prioritisation)")

    st.markdown(
        """
<div class="takeaway">
  <div class="takeaway-title">What this section is for</div>
  <div>It separates <b>big changes</b> from <b>noisy changes</b> so you don’t chase randomness.</div>
  <div class="takeaway-sub">Use the matrix as an action lens, then the confidence-interval view as an evidence lens.</div>
</div>
        """,
        unsafe_allow_html=True
    )

    st.plotly_chart(executive_impact_matrix(view), use_container_width=True)

    st.subheader("Effect sizes with confidence intervals (top 25)")
    st.plotly_chart(executive_forest_plot(view, top_n=min(25, len(view))), use_container_width=True)

st.divider()


# -----------------------------
# Decision table (not raw dump)
# -----------------------------
st.subheader("Decision table")

cols = []
for c in ["Month Year","Brand","Market","Category","KPI","Control Sample","Exposed Sample","Control_Pct","Exposed_Pct","Diff_PctPts","Lift_Pct","Reliability_Band","Clarity_Band","Notes_Short"]:
    if c in view.columns:
        cols.append(c)

decision = view[cols].copy()

# Clean formatting
if "Lift_Pct" in decision.columns:
    decision["Lift_Pct"] = decision["Lift_Pct"].round(1)
if "Diff_PctPts" in decision.columns:
    decision["Diff_PctPts"] = decision["Diff_PctPts"].round(2)
if "Control_Pct" in decision.columns:
    decision["Control_Pct"] = decision["Control_Pct"].round(1)
if "Exposed_Pct" in decision.columns:
    decision["Exposed_Pct"] = decision["Exposed_Pct"].round(1)

st.dataframe(decision, use_container_width=True)

st.divider()


# -----------------------------
# Export
# -----------------------------
st.subheader("Export")
if st.button("Generate PDF"):
    scope_df = view.copy() if export_scope == "All rows in view" else pd.DataFrame([row])
    pdf_bytes = build_pdf_bytes(scope_df, title=pdf_title)
    st.download_button("Download PDF", data=pdf_bytes, file_name=f"{pdf_title}.pdf", mime="application/pdf")
