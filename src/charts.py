import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go


# =============================
# PNG helper for PDF export
# =============================
def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# =============================
# Core helpers
# =============================
def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def _conf_score_from_p(pvals: pd.Series) -> pd.Series:
    """
    Certainty score for charts: -log10(p)
    Higher = more certain.
    Returns a pandas Series (NOT a numpy array) so .loc works.
    """
    p = pd.to_numeric(pvals, errors="coerce").astype(float)
    p = p.clip(lower=1e-12, upper=1.0)
    return (-np.log10(p)).astype(float)


def _fallback_reliability_band(df: pd.DataFrame, great=300, good=100, directional=50) -> pd.Series:
    """
    Fallback reliability band if Reliability_Band doesn't exist.
    Uses min(Control Sample, Exposed Sample).
    Returns pandas Series with df.index.
    """
    if "Control Sample" in df.columns and "Exposed Sample" in df.columns:
        n = pd.to_numeric(df["Control Sample"], errors="coerce").astype(float)
        m = pd.to_numeric(df["Exposed Sample"], errors="coerce").astype(float)
        min_n = pd.concat([n, m], axis=1).min(axis=1)
    else:
        min_n = pd.Series([np.nan] * len(df), index=df.index)

    def band(v):
        if not (v == v):
            return "Low"
        if v >= great:
            return "Great"
        if v >= good:
            return "Good"
        if v >= directional:
            return "Directional"
        return "Low"

    return min_n.apply(band)


def _fallback_clarity_band(df: pd.DataFrame, clear_p=0.05, directional_p=0.10, good_threshold=100) -> pd.Series:
    """
    Fallback clarity band if Clarity_Band doesn't exist.
    Uses p-value + reliability_n gate for Clear.
    Returns pandas Series with df.index.
    """
    if "P_Value" in df.columns:
        p = pd.to_numeric(df["P_Value"], errors="coerce").astype(float)
    else:
        p = pd.Series([np.nan] * len(df), index=df.index)

    # reliability_n gate
    if "Control Sample" in df.columns and "Exposed Sample" in df.columns:
        n = pd.to_numeric(df["Control Sample"], errors="coerce").astype(float)
        m = pd.to_numeric(df["Exposed Sample"], errors="coerce").astype(float)
        min_n = pd.concat([n, m], axis=1).min(axis=1)
    else:
        min_n = pd.Series([np.nan] * len(df), index=df.index)

    def band(pv, mn):
        if not (pv == pv):
            return "Unclear"
        if pv <= clear_p and (mn == mn and mn >= good_threshold):
            return "Clear"
        if pv <= directional_p:
            return "Directional"
        return "Unclear"

    return pd.Series([band(pv, mn) for pv, mn in zip(p.tolist(), min_n.tolist())], index=df.index)


def _ensure_bands(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures the dataframe has:
      - Reliability_Band  (Great/Good/Directional/Low)
      - Clarity_Band      (Clear/Directional/Unclear)

    Also attempts to map old Reliability values to Reliability_Band if present.
    """
    d = df.copy()

    # If a legacy 'Reliability' exists, map it into Reliability_Band (best-effort).
    if "Reliability_Band" not in d.columns and "Reliability" in d.columns:
        legacy = d["Reliability"].astype(str).str.strip().str.lower()

        mapping = {
            "high": "Great",
            "medium": "Good",
            "directional": "Directional",
            "low": "Low",
            "great": "Great",
            "good": "Good",
            "unclear": "Low",
        }
        d["Reliability_Band"] = legacy.map(mapping)

    if "Reliability_Band" not in d.columns:
        d["Reliability_Band"] = _fallback_reliability_band(d)

    if "Clarity_Band" not in d.columns:
        d["Clarity_Band"] = _fallback_clarity_band(d)

    return d


def _quadrant_from_effect(diff_pp, certainty, y_thr=1.301, x_thr=0.0):
    """
    Quadrants for Impact Matrix.
    x axis = effect size (Diff_PctPts preferred)
    y axis = certainty = -log10(p)
    """
    if diff_pp >= x_thr and certainty >= y_thr:
        return "Act"
    if diff_pp >= x_thr and certainty < y_thr:
        return "Watch"
    if diff_pp < x_thr and certainty >= y_thr:
        return "Investigate"
    return "Ignore"


QUAD_COLORS = {
    "Act": "rgba(22, 163, 74, 0.90)",          # green
    "Watch": "rgba(234, 179, 8, 0.90)",        # amber
    "Investigate": "rgba(239, 68, 68, 0.90)",  # red
    "Ignore": "rgba(148, 163, 184, 0.90)"      # grey
}

BAND_ORDER = ["Great", "Good", "Directional", "Low"]
CLARITY_ORDER = ["Clear", "Directional", "Unclear"]


# ============================================================
# MATPLOTLIB CHARTS (used by src/pdf_report.py for PDF export)
# ============================================================
def chart_control_vs_exposed_matplotlib(row):
    """
    PDF chart: control vs exposed bars.
    Expects row to contain Control_Pct and Exposed_Pct.
    """
    control = _safe_float(row.get("Control_Pct", np.nan))
    exposed = _safe_float(row.get("Exposed_Pct", np.nan))
    kpi = str(row.get("KPI", ""))
    brand = str(row.get("Brand", ""))

    fig = plt.figure(figsize=(6.0, 2.8))
    ax = fig.add_subplot(111)

    ax.bar(["Control", "Exposed"], [control, exposed])
    ax.set_ylabel("Score (%)")
    ax.set_title(f"{brand} — {kpi}")

    ax.text(0, control, f"{control:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.text(1, exposed, f"{exposed:.1f}%", ha="center", va="bottom", fontsize=9)

    lo = min(control, exposed) - 5
    hi = max(control, exposed) + 5
    if lo == lo and hi == hi:
        ax.set_ylim(lo, hi)

    ax.grid(True, axis="y", alpha=0.2)
    return fig


def chart_lift_rank_matplotlib(df, title="Lift by row (ranked)"):
    """
    PDF chart: ranked lift bars (horizontal).
    Requires Lift_Pct (or falls back to Diff_PctPts).
    """
    d = df.copy()

    if "Label" not in d.columns:
        brand = d["Brand"].astype(str) if "Brand" in d.columns else ""
        kpi = d["KPI"].astype(str) if "KPI" in d.columns else ""
        d["Label"] = brand + " • " + kpi

    if "Lift_Pct" not in d.columns:
        d["Lift_Pct"] = d.get("Diff_PctPts", 0.0)

    d = d.sort_values("Lift_Pct", ascending=True)

    fig = plt.figure(figsize=(7.0, max(2.8, 0.35 * len(d) + 1.8)))
    ax = fig.add_subplot(111)

    ax.barh(d["Label"], pd.to_numeric(d["Lift_Pct"], errors="coerce").astype(float))
    ax.set_xlabel("Relative lift (%)")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.2)
    return fig


def chart_confidence_quadrant_matplotlib(df, title="Effect vs certainty"):
    """
    PDF chart: scatter of effect vs certainty.
    Uses Diff_PctPts (preferred) and p-values.
    """
    d = df.copy()

    x = pd.to_numeric(d["Diff_PctPts"], errors="coerce").astype(float) if "Diff_PctPts" in d.columns \
        else pd.to_numeric(d.get("Lift_Pct", 0.0), errors="coerce").astype(float)

    y = _conf_score_from_p(d["P_Value"]) if "P_Value" in d.columns else pd.Series([0.0] * len(d), index=d.index)

    fig = plt.figure(figsize=(6.6, 4.2))
    ax = fig.add_subplot(111)

    ax.scatter(x, y)
    ax.axvline(0, alpha=0.2)
    ax.axhline(1.301, alpha=0.2)

    ax.set_xlabel("Effect (percentage points) — exposed minus control")
    ax.set_ylabel("Certainty (−log10 p)")
    ax.set_title(title)
    ax.grid(True, alpha=0.15)

    return fig


# =============================
# INTERACTIVE EXECUTIVE VISUALS
# =============================
def executive_reliability_ribbon(df):
    """
    Interactive bar: reliability mix.
    Uses Reliability_Band if present, else fallback.
    """
    d = _ensure_bands(df)

    counts = {k: int((d["Reliability_Band"] == k).sum()) for k in BAND_ORDER}

    fig = go.Figure(go.Bar(
        x=BAND_ORDER,
        y=[counts[k] for k in BAND_ORDER],
        hovertemplate="%{x}: %{y}<extra></extra>"
    ))

    fig.update_layout(
        title="Reliability mix",
        xaxis_title="Reliability",
        yaxis_title="Count of results",
        margin=dict(l=20, r=20, t=55, b=20),
        height=320
    )
    return fig


def executive_impact_matrix(df):
    """
    Interactive scatter: impact matrix.
    X = Diff_PctPts (preferred) else Lift_Pct
    Y = certainty = -log10(p)
    """
    d = _ensure_bands(df).copy()

    # --- Labels ---
    brand = d["Brand"].astype(str) if "Brand" in d.columns else pd.Series([""] * len(d), index=d.index)
    kpi = d["KPI"].astype(str) if "KPI" in d.columns else pd.Series([""] * len(d), index=d.index)
    month = d["Month Year"].astype(str) if "Month Year" in d.columns else pd.Series([""] * len(d), index=d.index)
    d["Label"] = np.where(month != "", brand + " • " + kpi + " • " + month, brand + " • " + kpi)

    # --- Axes as pandas Series (critical fix) ---
    x = pd.to_numeric(d["Diff_PctPts"], errors="coerce").astype(float) if "Diff_PctPts" in d.columns \
        else pd.to_numeric(d.get("Lift_Pct", 0.0), errors="coerce").astype(float)

    y = _conf_score_from_p(d["P_Value"]) if "P_Value" in d.columns else pd.Series([0.0] * len(d), index=d.index)

    y_thr = 1.301  # ~ p=0.05
    d["Certainty"] = y

    d["Quadrant"] = [
        _quadrant_from_effect(float(xx) if xx == xx else 0.0, float(yy) if yy == yy else 0.0, y_thr=y_thr, x_thr=0.0)
        for xx, yy in zip(x.tolist(), y.tolist())
    ]

    # Hover payload columns (Series with same index)
    def s_float(name, default=np.nan):
        if name in d.columns:
            return pd.to_numeric(d[name], errors="coerce").astype(float)
        return pd.Series([default] * len(d), index=d.index)

    control_pct = s_float("Control_Pct")
    exposed_pct = s_float("Exposed_Pct")
    diff_pp = s_float("Diff_PctPts")
    lift_pct = s_float("Lift_Pct")
    p_val = s_float("P_Value")

    rel = d["Reliability_Band"].astype(str)
    cla = d["Clarity_Band"].astype(str)

    fig = go.Figure()

    for q in ["Act", "Watch", "Investigate", "Ignore"]:
        subset = d[d["Quadrant"] == q]
        if subset.empty:
            continue

        idx = subset.index

        fig.add_trace(go.Scatter(
            x=x.loc[idx].astype(float),
            y=y.loc[idx].astype(float),
            mode="markers",
            name=q,
            marker=dict(size=11, color=QUAD_COLORS[q], line=dict(width=0)),
            text=subset["Label"],
            customdata=np.stack([
                control_pct.loc[idx].to_numpy(),
                exposed_pct.loc[idx].to_numpy(),
                diff_pp.loc[idx].to_numpy(),
                lift_pct.loc[idx].to_numpy(),
                p_val.loc[idx].to_numpy(),
                rel.loc[idx].to_numpy(),
                cla.loc[idx].to_numpy(),
            ], axis=1),
            hovertemplate=(
                "%{text}<br>"
                "Control: %{customdata[0]:.1f}% • Exposed: %{customdata[1]:.1f}%<br>"
                "Gap: %{customdata[2]:.2f} pts • Lift: %{customdata[3]:.1f}%<br>"
                "p: %{customdata[4]:.4f} • Reliability: %{customdata[5]} • Clarity: %{customdata[6]}"
                "<extra></extra>"
            )
        ))

    fig.add_hline(y=y_thr, line_width=1, opacity=0.25)
    fig.add_vline(x=0, line_width=1, opacity=0.25)

    fig.update_layout(
        title="Impact matrix",
        xaxis_title="Effect (percentage points) — right is positive, left is negative",
        yaxis_title="Certainty (−log10 p) — higher is stronger",
        legend_title="Quadrant",
        margin=dict(l=20, r=20, t=55, b=20),
        height=540
    )
    return fig


def executive_forest_plot(df, top_n=25):
    """
    Interactive effect-size view with confidence intervals.
    Uses Diff_PctPts + CI bands.
    """
    d = _ensure_bands(df).copy()

    # Label
    brand = d["Brand"].astype(str) if "Brand" in d.columns else pd.Series([""] * len(d), index=d.index)
    kpi = d["KPI"].astype(str) if "KPI" in d.columns else pd.Series([""] * len(d), index=d.index)
    month = d["Month Year"].astype(str) if "Month Year" in d.columns else pd.Series([""] * len(d), index=d.index)
    d["Label"] = np.where(month != "", brand + " • " + kpi + " • " + month, brand + " • " + kpi)

    if "Diff_PctPts" not in d.columns:
        # fallback
        d["Diff_PctPts"] = pd.to_numeric(d.get("Exposed_Pct", 0.0), errors="coerce") - pd.to_numeric(d.get("Control_Pct", 0.0), errors="coerce")

    d["AbsDiff"] = pd.to_numeric(d["Diff_PctPts"], errors="coerce").abs()
    d = d.sort_values("AbsDiff", ascending=False).head(int(min(top_n, len(d))))
    d = d.sort_values("Diff_PctPts", ascending=True)

    y_labels = d["Label"].tolist()
    diff = pd.to_numeric(d["Diff_PctPts"], errors="coerce").astype(float).tolist()

    have_ci = ("CI_Low_PctPts" in d.columns) and ("CI_High_PctPts" in d.columns)
    lo = pd.to_numeric(d["CI_Low_PctPts"], errors="coerce").astype(float).tolist() if have_ci else [np.nan] * len(d)
    hi = pd.to_numeric(d["CI_High_PctPts"], errors="coerce").astype(float).tolist() if have_ci else [np.nan] * len(d)

    fig = go.Figure()

    # CI lines
    for yy, l, h in zip(y_labels, lo, hi):
        if (l == l) and (h == h):
            fig.add_trace(go.Scatter(
                x=[l, h], y=[yy, yy],
                mode="lines",
                hoverinfo="skip",
                showlegend=False,
                opacity=0.6
            ))

    # markers
    fig.add_trace(go.Scatter(
        x=diff,
        y=y_labels,
        mode="markers",
        text=d["Reliability_Band"].astype(str),
        customdata=np.stack([d["Clarity_Band"].astype(str).to_numpy()], axis=1),
        hovertemplate="%{y}<br>Gap: %{x:.2f} pts<br>Reliability: %{text}<br>Clarity: %{customdata[0]}<extra></extra>",
        showlegend=False
    ))

    fig.add_vline(x=0, line_width=1, opacity=0.25)

    fig.update_layout(
        title=f"Effect sizes with confidence intervals (top {len(d)})",
        xaxis_title="Difference (percentage points) — exposed minus control",
        yaxis_title="",
        margin=dict(l=20, r=20, t=55, b=20),
        height=max(520, 22 * len(d) + 220)
    )
    return fig


def executive_story_card_chart(row):
    """
    Interactive story card for a single KPI row.
    """
    brand = str(row.get("Brand", ""))
    kpi = str(row.get("KPI", ""))

    control = _safe_float(row.get("Control_Pct", np.nan))
    exposed = _safe_float(row.get("Exposed_Pct", np.nan))
    diff = _safe_float(row.get("Diff_PctPts", np.nan))
    lift = _safe_float(row.get("Lift_Pct", np.nan))
    lo = _safe_float(row.get("CI_Low_PctPts", np.nan))
    hi = _safe_float(row.get("CI_High_PctPts", np.nan))
    p = _safe_float(row.get("P_Value", np.nan))

    rel = str(row.get("Reliability_Band", row.get("Reliability", "")))
    cla = str(row.get("Clarity_Band", ""))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Control", "Exposed"],
        y=[control, exposed],
        text=[f"{control:.1f}%", f"{exposed:.1f}%"],
        textposition="outside",
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>"
    ))

    subtitle_bits = []
    if diff == diff:
        subtitle_bits.append(f"Gap: {diff:.2f} pts")
    if lift == lift:
        subtitle_bits.append(f"Lift: {lift:.1f}%")
    if lo == lo and hi == hi:
        subtitle_bits.append(f"CI [{lo:.2f}, {hi:.2f}]")
    if p == p:
        subtitle_bits.append(f"p={p:.4f}")
    if rel:
        subtitle_bits.append(str(rel))
    if cla:
        subtitle_bits.append(str(cla))

    fig.update_layout(
        title=f"{brand} — {kpi}",
        yaxis_title="Score (%)",
        margin=dict(l=20, r=20, t=55, b=20),
        height=360
    )

    if subtitle_bits:
        fig.add_annotation(
            x=0.5, y=1.06, xref="paper", yref="paper",
            text=" • ".join(subtitle_bits),
            showarrow=False
        )
    return fig


# -----------------------------
# Backward-compatible names (if any legacy calls exist)
# -----------------------------
def interactive_lift_histogram(df):
    return executive_reliability_ribbon(df)

def interactive_confidence_scatter(df):
    return executive_impact_matrix(df)

def interactive_lift_rank(df):
    return executive_forest_plot(df, top_n=min(25, len(df)))

def interactive_dumbbell(row):
    return executive_story_card_chart(row)

def interactive_ci_interval(row):
    return executive_story_card_chart(row)
