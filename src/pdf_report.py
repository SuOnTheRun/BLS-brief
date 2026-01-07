import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

from .charts import chart_control_vs_exposed, chart_lift_rank, chart_confidence_quadrant, fig_to_png_bytes

def _draw_header(c, title, subtitle):
    c.setFont("Helvetica-Bold", 14)
    c.drawString(2.0*cm, 28.5*cm, title)

    c.setFont("Helvetica", 10)
    c.drawString(2.0*cm, 27.9*cm, subtitle)

    c.setLineWidth(0.4)
    c.line(2.0*cm, 27.6*cm, 19.5*cm, 27.6*cm)

def _draw_kpi_card(c, x, y, w, h, label, value, note):
    c.setLineWidth(0.6)
    c.rect(x, y, w, h)

    c.setFont("Helvetica", 9)
    c.drawString(x + 0.4*cm, y + h - 0.7*cm, label)

    c.setFont("Helvetica-Bold", 14)
    c.drawString(x + 0.4*cm, y + h - 1.6*cm, value)

    c.setFont("Helvetica", 8.5)
    c.drawString(x + 0.4*cm, y + 0.4*cm, note)

def build_pdf_bytes(filtered_df, cards, report_title, include_comparisons=True):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)

    dt = datetime.now().strftime("%d %b %Y")
    subtitle = f"Generated {dt} • Results shown with brand aliases"

    # Page 1
    _draw_header(c, report_title, subtitle)

    c.setFont("Helvetica", 10)
    c.drawString(2.0*cm, 26.8*cm, "Summary")

    total = len(filtered_df)
    sig = int(filtered_df["Significant_95"].sum()) if "Significant_95" in filtered_df.columns else 0
    avg_lift = filtered_df["Lift_Pct"].mean()

    _draw_kpi_card(
        c, 2.0*cm, 24.7*cm, 5.6*cm, 1.9*cm,
        "Rows in view", f"{total}", "Non-definitive rows may be included"
    )
    _draw_kpi_card(
        c, 8.0*cm, 24.7*cm, 5.6*cm, 1.9*cm,
        "Statistically clear", f"{sig}", "Clear = p-value below threshold"
    )
    _draw_kpi_card(
        c, 14.0*cm, 24.7*cm, 5.6*cm, 1.9*cm,
        "Average lift", f"{avg_lift:.2f}%", "Average across current view"
    )

    if include_comparisons and total > 1:
        tmp = filtered_df.copy()
        tmp["Label"] = tmp["Brand_Alias"].astype(str) + " • " + tmp["KPI"].astype(str)

        fig1 = chart_lift_rank(tmp[["Label", "Lift_Pct"]], title="Lift by row (ranked)")
        img1 = ImageReader(io.BytesIO(fig_to_png_bytes(fig1)))
        c.drawImage(img1, 2.0*cm, 14.6*cm, width=17.5*cm, height=8.6*cm, preserveAspectRatio=True, mask="auto")

        fig2 = chart_confidence_quadrant(filtered_df, title="Lift vs confidence")
        img2 = ImageReader(io.BytesIO(fig_to_png_bytes(fig2)))
        c.drawImage(img2, 2.0*cm, 7.0*cm, width=17.5*cm, height=7.0*cm, preserveAspectRatio=True, mask="auto")
    else:
        c.setFont("Helvetica", 9.5)
        c.drawString(2.0*cm, 23.8*cm, "Comparison charts are off, or there is only one row in view.")

    c.showPage()

    # One page per row (cap to avoid huge PDFs)
    max_pages = min(25, len(filtered_df))
    for i in range(max_pages):
        row = filtered_df.iloc[i]
        card = cards[i]

        title = f"{row['Brand_Alias']} — {row['KPI']} ({row.get('Month Year', '')})"
        _draw_header(c, title, subtitle)

        fig = chart_control_vs_exposed(row)
        img = ImageReader(io.BytesIO(fig_to_png_bytes(fig)))
        c.drawImage(img, 2.0*cm, 17.1*cm, width=17.5*cm, height=8.8*cm, preserveAspectRatio=True, mask="auto")

        c.setFont("Helvetica-Bold", 10)
        c.drawString(2.0*cm, 16.2*cm, card["state_label"])
        c.setFont("Helvetica", 9.5)
        c.drawString(2.0*cm, 15.7*cm, card["note"])

        c.setFont("Helvetica-Bold", 10)
        c.drawString(2.0*cm, 14.8*cm, "What changed")
        c.setFont("Helvetica", 9.5)
        c.drawString(2.0*cm, 14.3*cm, card["meaning"])

        c.setFont("Helvetica-Bold", 10)
        c.drawString(2.0*cm, 13.4*cm, "How to use it")
        c.setFont("Helvetica", 9.5)
        c.drawString(2.0*cm, 12.9*cm, card["decision"])

        c.setFont("Helvetica", 9)
        c.drawString(2.0*cm, 11.7*cm, f"Control: {row['Control_Pct']:.2f}%  (n={int(row['Control Sample'])})")
        c.drawString(2.0*cm, 11.2*cm, f"Exposed: {row['Exposed_Pct']:.2f}%  (n={int(row['Exposed Sample'])})")
        c.drawString(2.0*cm, 10.7*cm, f"Lift: {row['Lift_Pct']:.2f}%   •   Difference: {row['Diff_PctPts']:.2f} points")
        c.drawString(2.0*cm, 10.2*cm, f"p-value: {row['P_Value']:.4f}")

        c.showPage()

    c.save()
    buf.seek(0)
    return buf.getvalue()
