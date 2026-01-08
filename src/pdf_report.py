import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

from .charts import chart_control_vs_exposed_matplotlib, chart_lift_rank_matplotlib, chart_confidence_quadrant_matplotlib, fig_to_png_bytes

def _header(c, title, subtitle):
    c.setFont("Helvetica-Bold", 15)
    c.drawString(2.0 * cm, 28.4 * cm, title)
    c.setFont("Helvetica", 9.5)
    c.drawString(2.0 * cm, 27.8 * cm, subtitle)
    c.setLineWidth(0.6)
    c.line(2.0 * cm, 27.5 * cm, 19.5 * cm, 27.5 * cm)

def _stat_card(c, x, y, w, h, label, value, note):
    c.setLineWidth(0.6)
    c.roundRect(x, y, w, h, 8, stroke=1, fill=0)
    c.setFont("Helvetica", 8.8)
    c.drawString(x + 0.5 * cm, y + h - 0.7 * cm, label)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x + 0.5 * cm, y + h - 1.65 * cm, value)
    c.setFont("Helvetica", 8.4)
    c.drawString(x + 0.5 * cm, y + 0.45 * cm, note)

def build_pdf_bytes(filtered_df, cards, report_title, include_comparisons=True):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)

    dt = datetime.now().strftime("%d %b %Y")
    subtitle = f"Generated {dt}"

    # Page 1 — summary
    _header(c, report_title, subtitle)

    total = len(filtered_df)
    clear_count = int(filtered_df["Significant_95"].sum()) if "Significant_95" in filtered_df.columns else 0
    avg_lift = float(filtered_df["Lift_Pct"].mean()) if total else 0.0

    _stat_card(c, 2.0 * cm, 24.9 * cm, 5.6 * cm, 2.1 * cm, "Rows in view", f"{total}", "Current filtered view")
    _stat_card(c, 8.0 * cm, 24.9 * cm, 5.6 * cm, 2.1 * cm, "Statistically clear", f"{clear_count}", "p-value below threshold")
    _stat_card(c, 14.0 * cm, 24.9 * cm, 5.6 * cm, 2.1 * cm, "Average lift", f"{avg_lift:.2f}%", "Average across rows")

    c.setFont("Helvetica-Bold", 10.5)
    c.drawString(2.0 * cm, 24.0 * cm, "Highlights")

    # Highlights: top 5 absolute lift rows, plain language
    top = filtered_df.copy().sort_values("Lift_Pct", ascending=False).head(5)
    y = 23.5 * cm
    c.setFont("Helvetica", 9.4)
    for _, r in top.iterrows():
        status = "clear" if bool(r.get("Significant_95", False)) else "directional"
        line = f"{r.get('Brand','')} • {r.get('KPI','')}: {float(r.get('Lift_Pct',0)):.2f}% lift ({status})"
        c.drawString(2.0 * cm, y, line[:110])
        y -= 0.55 * cm
        if y < 21.0 * cm:
            break

    if include_comparisons and total > 1:
        tmp = filtered_df.copy()
        tmp["Label"] = tmp["Brand"].astype(str) + " • " + tmp["KPI"].astype(str)

        fig1 = chart_lift_rank_matplotlib(tmp[["Label", "Lift_Pct"]], title="Lift by row (ranked)")
        img1 = ImageReader(io.BytesIO(fig_to_png_bytes(fig1)))
        c.drawImage(img1, 2.0 * cm, 12.9 * cm, width=17.5 * cm, height=7.8 * cm, preserveAspectRatio=True, mask="auto")

        fig2 = chart_confidence_quadrant_matplotlib(filtered_df, title="Lift vs confidence")
        img2 = ImageReader(io.BytesIO(fig_to_png_bytes(fig2)))
        c.drawImage(img2, 2.0 * cm, 5.4 * cm, width=17.5 * cm, height=7.0 * cm, preserveAspectRatio=True, mask="auto")

    c.showPage()

    # Deep dives (cap)
    max_pages = min(25, len(filtered_df))
    for i in range(max_pages):
        row = filtered_df.iloc[i]
        card = cards[i]

        title = f"{row.get('Brand','')} — {row.get('KPI','')} ({row.get('Month Year','')})"
        _header(c, title, subtitle)

        fig = chart_control_vs_exposed_matplotlib(row)
        img = ImageReader(io.BytesIO(fig_to_png_bytes(fig)))
        c.drawImage(img, 2.0 * cm, 17.2 * cm, width=17.5 * cm, height=8.6 * cm, preserveAspectRatio=True, mask="auto")

        c.setFont("Helvetica-Bold", 10.2)
        c.drawString(2.0 * cm, 16.2 * cm, card["state_label"])

        c.setFont("Helvetica", 9.4)
        c.drawString(2.0 * cm, 15.7 * cm, str(card["note"])[:110])

        c.setFont("Helvetica-Bold", 10.0)
        c.drawString(2.0 * cm, 14.7 * cm, "What changed")
        c.setFont("Helvetica", 9.4)
        c.drawString(2.0 * cm, 14.2 * cm, str(card["meaning"])[:110])

        c.setFont("Helvetica-Bold", 10.0)
        c.drawString(2.0 * cm, 13.3 * cm, "How to use it")
        c.setFont("Helvetica", 9.4)
        c.drawString(2.0 * cm, 12.8 * cm, str(card["decision"])[:110])

        c.setFont("Helvetica", 9.2)
        c.drawString(2.0 * cm, 11.6 * cm, f"Control: {row['Control_Pct']:.2f}% (n={int(row['Control Sample'])})")
        c.drawString(2.0 * cm, 11.1 * cm, f"Exposed: {row['Exposed_Pct']:.2f}% (n={int(row['Exposed Sample'])})")
        c.drawString(2.0 * cm, 10.6 * cm, f"Lift: {row['Lift_Pct']:.2f}%   •   Difference: {row['Diff_PctPts']:.2f} pts")
        c.drawString(2.0 * cm, 10.1 * cm, f"p-value: {row['P_Value']:.4f}")

        c.showPage()

    c.save()
    buf.seek(0)
    return buf.getvalue()
