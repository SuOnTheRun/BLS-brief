import io
import numpy as np
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
        v = float(x)
        return v
    except Exception:
        return default


def _conf_score_from_p(pvals):
    """
    Certainty score for charts: -log10(p)
    Higher = more certain.
    """
    p = np.array(pvals, dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    return -np.log10(p)


def _fallback_reliability_band(df, great=300, good=100, directional=50):
    """
    Fallback reliability band if Reliability_Band doesn't exist.
    Uses min(Control Sample, Exposed Sample).
    """
    if "Control Sample" in df.columns and "Exposed Sample" in df.columns:
        n = np.minimum(
            df["Control Sample"].astype(float).to_numpy(),
            df["Exposed Sample"].astype(float).to_numpy(),
        )
    else:
        # If samples missing, treat as unknown -> Low
        n = np.full(len(df), np.nan)

    out = []
    for v in n:
        if not (v == v):
            out.append("Low")
        elif v >= great:
            out.append("Great")
        elif v >= good:
            out.append("Good")
        elif v >= directional:
            out.append("Directional")
        else:
            out.append("Low")
    return np.array(out, dtype=object)


def _fallback_clarity_band(df, clear_p=0.05, directional_p=0.10, good_threshold=100):
    """
    Fallback clarity band if Clarity_Band doesn't exist.
    Uses p-value + reliability_n gate (min sample >= good_threshold) for Clear.
    """
    p = df["P_Value"].astype(float).to_numpy() if "P_Value" in df.columns else np.full(len(df), np.nan)

    # reliability_n gate
    if "Control Sample" in df.columns and "Exposed Sample" in df.columns:
        min_n = np.minimum(
            df["Control Sample"].astype(float).to_numpy(),
            df["Exposed Sample"].astype(float).to_numpy(),
        )
    else:
        min_n = np.full(len(df), np.nan)

    out = []
    for pv, mn in zip(p, min_n):
        if not (pv == pv):
            out.append("Unclear")
            continue
        if pv <= clear_p and (mn == mn and mn >= good_threshold):
            out.append("Clear")
        elif pv <= directional_p:
            out.append("Directional")
        else:
            out.append("Unclear")
    return np.array(out, dtype=object)


def _ensure_bands(df):
    """
    Ensures the dataframe has:
      - Reliability_Band  (Great/Good/Directional/Low)
      - Clarity_Band      (Clear/Directional/Unclear)

    If not present, creates fallbacks.
    """
    d = df.copy()

    if "Reliability_Band" not in d.columns:
        d["Reliability_Band"] = _fallback_reliability_band(d)

    if "Clarity_Band" not in d.columns:
        d["Clarity_Band"] = _fallback_clarity_band(d)

    return d


def _quadrant_from_effect(diff_pp, certainty, y_thr=1.301, x_thr=0.0):
    """
    Quadrants for Impact Matrix.
    - x axis is effect size (Diff_PctPts by default): right positive, left negative
    - y axis is certainty: higher stronger
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
    Requires Lift_Pct, Brand, KPI.
    """
    d = df.copy()

    if "Label" not in d.columns:
        d["Label"] = d.get("Brand", "").astype(str) + " • " + d.get("KPI", "").astype(str)

    if "Lift_Pct" not in d.columns:
        # If lift isn't present, fall back to Diff_PctPts
        d["Lift_Pct"] = d.get("Diff_PctPts", 0.0)

    d = d.sort_values("Lift_Pct", ascending=True)

    fig = plt.figure(figsize=(7.0, max(2.8, 0.35 * len(d) + 1.8)))
    ax = fig.add_subplot(111)

    ax.barh(d["Label"], d["Lift_Pct"].astype(float))
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

    x = d["Diff_PctPts"].astype(float) if "Diff_PctPts" in d.columns else d["Lift_Pct"].astype(float)
    y = _conf_score_from_p(d["P_Value"].astype(float)) if "P_Value" in d.columns else np.zeros(len(d))

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


def executive_clarity_mix(df):
    """
    Interactive bar: clarity mix (Clear / Directional / Unclear).
    """
    d = _ensure_bands(df)
    counts = {k: int((d["Clarity_Band"] == k).sum()) for k in CLARITY_ORDER}

    fig = go.Figure(go.Bar(
        x=CLARITY_ORDER,
        y=[counts[k] for k in CLARITY_ORDER],
        hovertemplate="%{x}: %{y}<extra></extra>"
    ))

    fig.update_layout(
        title="Statistical clarity mix",
        xaxis_title="Clarity",
        yaxis_title="Count of results",
        margin=dict(l=20, r=20, t=55, b=20),
        height=320
    )
    return fig


def executive_impact_matrix(df):
    """
    Interactive scatter: impact matrix.
    Best practice here is to use Diff_PctPts on x-axis (clearer than Lift when baseline is small).
    y-axis is certainty = -log10(p).
    Dots are colored by quadrant.
    """
    d = _ensure_bands(df).copy()

    # Labels
    brand = d["Brand"].astype(str) if "Brand" in d.columns else ""
    kpi = d["KPI"].astype(str) if "KPI" in d.columns else ""
    month = d["Month Year"].astype(str) if "Month Year" in d.columns else ""
    d["Label"] = np.where(month != "", brand + " • " + kpi + " • " + month, brand + " • " + kpi)

    # Axes
    x = d["Diff_PctPts"].astype(float) if "Diff_PctPts" in d.columns else d["Lift_Pct"].astype(float)
    y = _conf_score_from_p(d["P_Value"].astype(float)) if "P_Value" in d.columns else np.zeros(len(d))

    y_thr = 1.301  # ~ p=0.05
    d["Certainty"] = y

    d["Quadrant"] = [
        _quadrant_from_effect(float(xx), float(yy), y_thr=y_thr, x_thr=0.0)
        for xx, yy in zip(x.tolist(), y.tolist())
    ]

    # Hover payload
    def col_or_nan(name):
        return d[name].astype(float) if name in d.columns else np.full(len(d), np.nan)

    control_pct = col_or_nan("Control_Pct")
    exposed_pct = col_or_nan("Exposed_Pct")
    diff_pp = col_or_nan("Diff_PctPts")
    lift_pct = col_or_nan("Lift_Pct")
    p_val = col_or_nan("P_Value")

    # Reliability + clarity for hover
    rel = d["Reliability_Band"].astype(str)
    cla = d["Clarity_Band"].astype(str)

    fig = go.Figure()

    for q in ["Act", "Watch", "Investigate", "Ignore"]:
        subset = d[d["Quadrant"] == q]
        if len(subset) == 0:
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
                control_pct.loc[idx],
                exposed_pct.loc[idx],
                diff_pp.loc[idx],
                lift_pct.loc[idx],
                p_val.loc[idx],
                rel.loc[idx],
                cla.loc[idx]
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

    # Build label
    brand = d["Brand"].astype(str) if "Brand" in d.columns else ""
    kpi = d["KPI"].astype(str) if "KPI" in d.columns else ""
    month = d["Month Year"].astype(str) if "Month Year" in d.columns else ""
    d["Label"] = np.where(month != "", brand + " • " + kpi + " • " + month, brand + " • " + kpi)

    if "Diff_PctPts" not in d.columns:
        d["Diff_PctPts"] = d.get("Exposed_Pct", 0.0) - d.get("Control_Pct", 0.0)

    d["AbsDiff"] = d["Diff_PctPts"].astype(float).abs()
    d = d.sort_values("AbsDiff", ascending=False).head(int(min(top_n, len(d))))
    d = d.sort_values("Diff_PctPts", ascending=True)

    y = d["Label"].tolist()
    diff = d["Diff_PctPts"].astype(float).tolist()

    if "CI_Low_PctPts" in d.columns and "CI_High_PctPts" in d.columns:
        lo = d["CI_Low_PctPts"].astype(float).tolist()
        hi = d["CI_High_PctPts"].astype(float).tolist()
    else:
        # Fallback if CI missing
        lo = [np.nan for _ in diff]
        hi = [np.nan for _ in diff]

    fig = go.Figure()

    # CI lines
    for yy, l, h in zip(y, lo, hi):
        if (l == l) and (h == h):
            fig.add_trace(go.Scatter(
                x=[l, h], y=[yy, yy],
                mode="lines",
                hoverinfo="skip",
                showlegend=False,
                opacity=0.6
            ))

    # Point markers
    fig.add_trace(go.Scatter(
        x=diff,
        y=y,
        mode="markers",
        text=d["Reliability_Band"].astype(str),
        customdata=np.stack([d["Clarity_Band"].astype(str)], axis=1),
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
    Assumes row contains Control_Pct, Exposed_Pct, Diff_PctPts, P_Value (optional CI).
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
# Backward-compatible names (app.py may use these)
# -----------------------------

def interactive_lift_histogram(df):
    # Historically used for "overview" – now points to reliability mix
    return executive_reliability_ribbon(df)

def interactive_confidence_scatter(df):
    return executive_impact_matrix(df)

def interactive_lift_rank(df):
    return executive_forest_plot(df, top_n=min(25, len(df)))

def interactive_dumbbell(row):
    return executive_story_card_chart(row)

def interactive_ci_interval(row):
    return executive_story_card_chart(row)
