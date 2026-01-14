import io
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# =============================
# Helpers
# =============================
def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def _conf_score(pvals):
    p = np.clip(np.array(pvals, dtype=float), 1e-12, 1.0)
    return -np.log10(p)

def _quadrant(lift, certainty, y_thr=1.301):
    if lift >= 0 and certainty >= y_thr:
        return "Act"
    if lift >= 0 and certainty < y_thr:
        return "Watch"
    if lift < 0 and certainty >= y_thr:
        return "Investigate"
    return "Ignore"

QUAD_COLORS = {
    "Act": "rgba(22, 163, 74, 0.9)",         # green
    "Watch": "rgba(234, 179, 8, 0.9)",       # amber
    "Investigate": "rgba(239, 68, 68, 0.9)", # red
    "Ignore": "rgba(148, 163, 184, 0.9)"     # grey
}

# ============================================================
# MATPLOTLIB CHARTS (used by src/pdf_report.py for PDF export)
# ============================================================
def chart_control_vs_exposed_matplotlib(row):
    """
    A simple but clean "control vs exposed" chart for PDFs.
    Expects row to contain Control_Pct and Exposed_Pct (in % scale, e.g., 47.1).
    """
    control = float(row["Control_Pct"])
    exposed = float(row["Exposed_Pct"])
    kpi = str(row.get("KPI", ""))
    brand = str(row.get("Brand", ""))

    fig = plt.figure(figsize=(6.0, 2.8))
    ax = fig.add_subplot(111)

    ax.bar(["Control", "Exposed"], [control, exposed])
    ax.set_ylabel("Score (%)")
    ax.set_title(f"{brand} — {kpi}")

    ax.text(0, control, f"{control:.2f}%", ha="center", va="bottom", fontsize=9)
    ax.text(1, exposed, f"{exposed:.2f}%", ha="center", va="bottom", fontsize=9)

    lo = min(control, exposed) - 5
    hi = max(control, exposed) + 5
    ax.set_ylim(lo, hi)
    ax.grid(True, axis="y", alpha=0.2)
    return fig

def chart_lift_rank_matplotlib(df, title="Lift by row (ranked)"):
    """
    PDF chart: ranked lift bars (horizontal).
    Requires df columns: Lift_Pct, Brand, KPI (optional Label).
    """
    d = df.copy()
    if "Label" not in d.columns:
        d["Label"] = d["Brand"].astype(str) + " • " + d["KPI"].astype(str)

    d = d.sort_values("Lift_Pct", ascending=True)

    fig = plt.figure(figsize=(7.0, max(2.8, 0.35 * len(d) + 1.8)))
    ax = fig.add_subplot(111)
    ax.barh(d["Label"], d["Lift_Pct"].astype(float))
    ax.set_xlabel("Relative lift (%)")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.2)
    return fig

def chart_confidence_quadrant_matplotlib(df, title="Lift vs confidence"):
    """
    PDF chart: scatter of lift vs certainty.
    Requires columns: Lift_Pct, P_Value
    """
    d = df.copy()
    x = d["Lift_Pct"].astype(float)
    y = _conf_score(d["P_Value"].astype(float))

    fig = plt.figure(figsize=(6.6, 4.2))
    ax = fig.add_subplot(111)
    ax.scatter(x, y)
    ax.axvline(0, alpha=0.2)
    ax.axhline(1.301, alpha=0.2)
    ax.set_xlabel("Relative lift (%)")
    ax.set_ylabel("-log10(p-value)")
    ax.set_title(title)
    ax.grid(True, alpha=0.15)
    return fig

# =============================
# INTERACTIVE EXECUTIVE VISUALS
# =============================
def executive_reliability_ribbon(df):
    order = ["High", "Medium", "Directional", "Low"]
    counts = {k: int((df["Reliability"] == k).sum()) for k in order}

    fig = go.Figure(go.Bar(
        x=order,
        y=[counts[k] for k in order],
        hovertemplate="%{x}: %{y}<extra></extra>"
    ))
    fig.update_layout(
        title="Reliability mix (how much you can trust the view)",
        xaxis_title="Reliability",
        yaxis_title="Count of rows",
        margin=dict(l=20, r=20, t=55, b=20),
        height=320
    )
    return fig

def executive_impact_matrix(df):
    tmp = df.copy()
    tmp["Label"] = tmp["Brand"].astype(str) + " • " + tmp["KPI"].astype(str)
    tmp["Certainty"] = _conf_score(tmp["P_Value"])

    y_thr = 1.301  # ~ p=0.05
    tmp["Quadrant"] = [
        _quadrant(float(x), float(y), y_thr=y_thr)
        for x, y in zip(tmp["Lift_Pct"].astype(float), tmp["Certainty"].astype(float))
    ]

    fig = go.Figure()
    for q in ["Act", "Watch", "Investigate", "Ignore"]:
        d = tmp[tmp["Quadrant"] == q]
        if len(d) == 0:
            continue

        fig.add_trace(go.Scatter(
            x=d["Lift_Pct"].astype(float),
            y=d["Certainty"].astype(float),
            mode="markers",
            name=q,
            marker=dict(size=10, color=QUAD_COLORS[q]),
            text=d["Label"],
            customdata=np.stack([d["Diff_PctPts"], d["P_Value"], d["Reliability"]], axis=1),
            hovertemplate=(
                "%{text}<br>"
                "Lift: %{x:.2f}%<br>"
                "Gap: %{customdata[0]:.2f} pts<br>"
                "p: %{customdata[1]:.4f}<br>"
                "Reliability: %{customdata[2]}<extra></extra>"
            )
        ))

    fig.add_hline(y=y_thr, line_width=1, opacity=0.25)
    fig.add_vline(x=0, line_width=1, opacity=0.25)

    fig.add_annotation(x=0.78, y=0.96, xref="paper", yref="paper", text="Act", showarrow=False)
    fig.add_annotation(x=0.78, y=0.07, xref="paper", yref="paper", text="Watch", showarrow=False)
    fig.add_annotation(x=0.12, y=0.96, xref="paper", yref="paper", text="Investigate", showarrow=False)
    fig.add_annotation(x=0.12, y=0.07, xref="paper", yref="paper", text="Ignore", showarrow=False)

    fig.update_layout(
        title="Impact matrix (what to act on)",
        xaxis_title="Lift (%) — right is positive, left is negative",
        yaxis_title="Certainty (−log10 p) — higher is stronger",
        legend_title="Quadrant",
        margin=dict(l=20, r=20, t=55, b=20),
        height=520
    )
    return fig

def executive_forest_plot(df, top_n=25):
    tmp = df.copy()
    tmp["Label"] = tmp["Brand"].astype(str) + " • " + tmp["KPI"].astype(str) + " • " + tmp["Month Year"].astype(str)

    tmp["AbsDiff"] = tmp["Diff_PctPts"].abs()
    tmp = tmp.sort_values("AbsDiff", ascending=False).head(top_n)
    tmp = tmp.sort_values("Diff_PctPts", ascending=True)

    y = tmp["Label"].tolist()
    diff = tmp["Diff_PctPts"].astype(float).tolist()
    lo = tmp["CI_Low_PctPts"].astype(float).tolist()
    hi = tmp["CI_High_PctPts"].astype(float).tolist()

    fig = go.Figure()

    for yy, l, h in zip(y, lo, hi):
        fig.add_trace(go.Scatter(
            x=[l, h], y=[yy, yy],
            mode="lines",
            hoverinfo="skip",
            showlegend=False,
            opacity=0.6
        ))

    fig.add_trace(go.Scatter(
        x=diff, y=y,
        mode="markers",
        text=tmp["Reliability"].astype(str),
        hovertemplate="%{y}<br>Diff: %{x:.2f} pts<br>Reliability: %{text}<extra></extra>",
        showlegend=False
    ))

    fig.add_vline(x=0, line_width=1, opacity=0.25)

    fig.update_layout(
        title=f"Effect sizes with confidence intervals (top {len(tmp)})",
        xaxis_title="Difference (percentage points) — exposed minus control",
        yaxis_title="",
        margin=dict(l=20, r=20, t=55, b=20),
        height=max(520, 22 * len(tmp) + 220)
    )
    return fig

def executive_story_card_chart(row):
    brand = str(row.get("Brand", ""))
    kpi = str(row.get("KPI", ""))

    control = float(row["Control_Pct"])
    exposed = float(row["Exposed_Pct"])
    diff = float(row["Diff_PctPts"])
    lo = float(row.get("CI_Low_PctPts", np.nan))
    hi = float(row.get("CI_High_PctPts", np.nan))
    p = float(row["P_Value"])
    rel = str(row.get("Reliability", ""))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Control", "Exposed"],
        y=[control, exposed],
        text=[f"{control:.2f}%", f"{exposed:.2f}%"],
        textposition="outside",
        hovertemplate="%{x}: %{y:.2f}%<extra></extra>"
    ))

    fig.update_layout(
        title=f"{brand} — {kpi}",
        yaxis_title="Score (%)",
        margin=dict(l=20, r=20, t=55, b=20),
        height=360
    )

    fig.add_annotation(
        x=0.5, y=1.05, xref="paper", yref="paper",
        text=f"Gap: {diff:.2f} pts • CI [{lo:.2f}, {hi:.2f}] • p={p:.4f} • {rel}",
        showarrow=False
    )
    return fig

# -----------------------------
# Backward-compatible names (app.py uses these)
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
