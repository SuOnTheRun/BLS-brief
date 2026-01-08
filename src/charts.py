import io
import numpy as np
import matplotlib.pyplot as plt

def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def chart_control_vs_exposed(row):
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

def chart_lift_rank(df, title="Lift by KPI"):
    fig = plt.figure(figsize=(7.0, max(2.6, 0.35*len(df) + 1.8)))
    ax = fig.add_subplot(111)

    df2 = df.copy().sort_values("Lift_Pct", ascending=True)
    ax.barh(df2["Label"], df2["Lift_Pct"])
    ax.set_xlabel("Relative lift (%)")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.2)
    return fig

def chart_confidence_quadrant(df, title="Lift vs confidence"):
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
