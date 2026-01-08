import io
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# -----------------------------
# Helpers for PDF (matplotlib)
# -----------------------------
def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# =============================
# Matplotlib charts (PDF)
# =============================
def chart_control_vs_exposed_matplotlib(row):
    control = float(row["Control_Pct"])
    exposed = float(row["Exposed_Pct"])
    kpi = str(row["KPI"])
    brand = str(row.get("Brand", ""))

    fig = plt.figure(figsize=(6.2, 2.6))
    ax = fig.add_subplot(111)

    ax.scatter([0, 1], [control, exposed])
    ax.plot([0, 1], [control, exposed])

    ax.set_xticks([0, 1], ["Control", "Exposed"])
    ax.set_ylabel("Score (%)")
    ax.set_title(f"{brand} — {kpi}")

    ax.text(0, control, f" {control:.2f}%", va="bottom")
    ax.text(1, exposed, f" {exposed:.2f}%", va="bottom")

    ax.set_ylim(min(control, exposed) - 5, max(control, exposed) + 5)
    ax.grid(True, axis="y", alpha=0.2)
    return fig

def chart_lift_rank_matplotlib(df, title="Lift by row (ranked)"):
    fig = plt.figure(figsize=(7.0, max(2.6, 0.35 * len(df) + 1.8)))
    ax = fig.add_subplot(111)

    df2 = df.copy().sort_values("Lift_Pct", ascending=True)
    ax.barh(df2["Label"], df2["Lift_Pct"])
    ax.set_xlabel("Relative lift (%)")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.2)
    return fig

def chart_confidence_quadrant_matplotlib(df, title="Lift vs confidence"):
    fig = plt.figure(figsize=(6.6, 4.0))
    ax = fig.add_subplot(111)

    x = df["Lift_Pct"].astype(float)
    p = df["P_Value"].astype(float)
    y = -np.log10(np.clip(p, 1e-12, 1.0))

    ax.scatter(x, y)
    ax.axvline(0, alpha=0.2)
    ax.set_xlabel("Relative lift (%)")
    ax.set_ylabel("-log10(p-value)")
    ax.set_title(title)
    ax.grid(True, alpha=0.15)
    return fig

# =============================
# Plotly charts (interactive UI)
# =============================
def interactive_dumbbell(row):
    brand = str(row.get("Brand", ""))
    kpi = str(row.get("KPI", ""))
    control = float(row["Control_Pct"])
    exposed = float(row["Exposed_Pct"])
    diff = float(row["Diff_PctPts"])
    pval = float(row["P_Value"])
    sig = bool(row["Significant_95"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[control, exposed],
        y=["Score", "Score"],
        mode="markers+lines",
        text=["Control", "Exposed"],
        hovertemplate="Group: %{text}<br>Score: %{x:.2f}%<extra></extra>"
    ))

    fig.update_layout(
        title=f"{brand} — {kpi}",
        xaxis_title="Score (%)",
        yaxis=dict(showticklabels=False),
        margin=dict(l=20, r=20, t=50, b=20),
        height=260
    )

    fig.add_annotation(
        x=exposed, y=1, xref="x", yref="paper",
        text=f"Gap: {diff:.2f} pts • p={pval:.4f} • {'clear' if sig else 'directional'}",
        showarrow=False, yanchor="bottom"
    )
    return fig

def interactive_lift_rank(df):
    tmp = df.copy()
    tmp["Label"] = tmp["Brand"].astype(str) + " • " + tmp["KPI"].astype(str)
    tmp = tmp.sort_values("Lift_Pct", ascending=True)

    fig = go.Figure(go.Bar(
        x=tmp["Lift_Pct"],
        y=tmp["Label"],
        orientation="h",
        hovertemplate="Lift: %{x:.2f}%<br>%{y}<extra></extra>"
    ))

    fig.update_layout(
        title="Lift by row (ranked)",
        xaxis_title="Relative lift (%)",
        margin=dict(l=20, r=20, t=50, b=20),
        height=max(360, 28 * len(tmp) + 140)
    )
    return fig

def interactive_confidence_scatter(df):
    tmp = df.copy()
    tmp["Label"] = tmp["Brand"].astype(str) + " • " + tmp["KPI"].astype(str)
    tmp["Conf"] = -np.log10(np.clip(tmp["P_Value"].astype(float), 1e-12, 1.0))

    fig = go.Figure(go.Scatter(
        x=tmp["Lift_Pct"].astype(float),
        y=tmp["Conf"],
        mode="markers",
        text=tmp["Label"],
        hovertemplate="%{text}<br>Lift: %{x:.2f}%<br>-log10(p): %{y:.2f}<extra></extra>"
    ))

    fig.add_vline(x=0, line_width=1)

    fig.update_layout(
        title="Lift vs confidence",
        xaxis_title="Relative lift (%)",
        yaxis_title="-log10(p-value)",
        margin=dict(l=20, r=20, t=50, b=20),
        height=420
    )
    return fig

def interactive_lift_histogram(df):
    x = df["Lift_Pct"].astype(float)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=x,
        nbinsx=18,
        hovertemplate="Lift bin count: %{y}<extra></extra>"
    ))
    fig.add_vline(x=0, line_width=1)
    fig.update_layout(
        title="Lift distribution",
        xaxis_title="Lift (%)",
        yaxis_title="Count of rows",
        margin=dict(l=20, r=20, t=50, b=20),
        height=320
    )
    return fig

def interactive_ci_interval(row):
    brand = str(row.get("Brand", ""))
    kpi = str(row.get("KPI", ""))

    diff = float(row["Diff_PctPts"])
    lo = float(row.get("CI_Low_PctPts", np.nan))
    hi = float(row.get("CI_High_PctPts", np.nan))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[lo, hi],
        y=["Difference", "Difference"],
        mode="lines",
        hovertemplate="CI: %{x:.2f} pts<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=[diff],
        y=["Difference"],
        mode="markers",
        hovertemplate="Diff: %{x:.2f} pts<extra></extra>"
    ))
    fig.add_vline(x=0, line_width=1)

    fig.update_layout(
        title=f"Difference range (CI) — {brand} • {kpi}",
        xaxis_title="Difference (percentage points)",
        yaxis=dict(showticklabels=False),
        margin=dict(l=20, r=20, t=50, b=20),
        height=240
    )
    return fig
