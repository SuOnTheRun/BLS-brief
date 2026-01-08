import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

from .charts import chart_control_vs_exposed, chart_lift_rank, chart_confidence_quadrant, fig_to_png_bytes

def _header(c, title, subtitle):
    c.setFont("Helvetica-Bold", 15)
    c.drawString(2.0*cm, 28.4*cm, title)
    c.setFont("Helvetica", 9.5)
    c.drawString(2.0*cm, 27.8*cm, subtitle)
    c.setLineWidth(0.6)
    c.line(2.0*cm, 27.5*cm, 19.5*cm, 27.5*cm)

def _stat_card(c, x, y, w, h, label, value, note):
    c.setLineWidth(0.6)
    c.roundRect(x, y, w, h, 8, stroke=1, fill=0)
    c.setFont("Helvetica", 8.8)
    c.drawString(x + 0.5*cm, y + h - 0.7*cm, label)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x + 0.5*cm, y + h - 1.65*cm, value)
    c.setFont("Helvetica", 8.4)
    c.drawString(x + 0.5*cm, y + 0.45*cm, note)

def build_pdf_bytes(filtered_df, cards, report_title, include_comparisons=True):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)

    dt = datetime.now().strftime("%d %b %Y")
    subtitle = f"Generated {dt}"

    # Page 1: summary
    _header(c, report_title, subtitle)

    total = len(filtered_df)
    clear_count = int(filtered_df["Significant_95"].sum()) if "Significant_95" in filtered_df.columns else 0
    avg_lift = float(filtered_df["Lift_Pct"].mean()) if total else 0.0

    _stat_card(c, 2.0*cm, 24.9*cm, 5.6*cm, 2.1*cm, "Rows in view", f"{total}", "Current filtered view")
    _stat_card(c, 8.0*cm, 24.9*cm, 5.6*cm, 2.1*cm, "Statistically clear", f"{clear_count}", "p-value below threshold")
    _stat_card(c, 14.0*cm, 24.9*cm, 5.6*cm, 2.1*cm, "Average lift", f"{avg_lift:.2f}%", "Average across rows")

    c.setFont("Helvetica-Bold", 10.5)
    c.drawString(2.0*cm, 24.0*cm, "Highlights")

    # Keep highlights plain + non-hype: top 5 absolute lifts
    top = filtered_df.copy().sort_values("Lift_Pct", ascending=False).head(5)
    y = 23.5*cm
    c.setFont("Helvetica", 9.4)
    for _, r in top.iterrows():
        line = f"{r.get('Brand','')} • {r.get('KPI','')}: {r.get('Lift_Pct',0):.2f}% lift ({'clear' if r.get('Significant_95',False) else 'directional'})"
        c.drawString(2.0*cm, y, line[:110])
        y -= 0.55*cm
        if y < 21.0*cm:
            break

    # Optional comparisons
    if include_comparisons and total > 1:
        tmp = filtered_df.copy()
        tmp["Label"] = tmp["Brand"].astype(str) + " • " + tmp["KPI"].astype(str)

        fig1 = chart_lift_rank(tmp[["Label", "Lift_Pct"]], title="Lift by row (ranked)")
        img1 = ImageReader(io.BytesIO(fig_to_png_bytes(fig1)))
        c.drawImage(img1, 2.0*cm, 12.9*cm, width=17.5*cm, height=7.8*cm, preserveAspectRatio=True, mask="auto")

        fig2 = chart_confidence_quadrant(filtered_df, title="Lift vs confidence")
        img2 = ImageReader(io.BytesIO(fig_to_png_bytes(fig2)))
        c.drawImage(img2, 2.0*cm, 5.4*cm, width=17.5*cm, height=7.0*cm, preserveAspectRatio=True, mask="auto")

    c.showPage()

    # Deep dives: clean one-page per row (cap)
    max_pages = min(25, len(filtered_df))
    for i in range(max_pages):
        row = filtered_df.iloc[i]
        card = cards[i]

        title = f"{row.get('Brand','')} — {row.get('KPI','')} ({row.get('Month Year','')})"
        _header(c, title, subtitle)

        fig = chart_control_vs_exposed(row)
        img = ImageReader(io.BytesIO(fig_to_png_bytes(fig)))
        c.drawImage(img, 2.0*cm, 17.2*cm, width=17.5*cm, height=8.6*cm, preserveAspectRatio=True, mask="auto")

        # Text blocks (plain)
        c.setFont("Helvetica-Bold", 10.2)
        c.drawString(2.0*cm, 16.2*cm, card["state_label"])

        c.setFont("Helvetica", 9.4)
        c.drawString(2.0*cm, 15.7*cm, card["note"][:110])

        c.setFont("Helvetica-Bold", 10.0)
        c.drawString(2.0*cm, 14.7*cm, "What changed")
        c.setFont("Helvetica", 9.4)
        c.drawString(2.0*cm, 14.2*cm, card["meaning"][:110])

        c.setFont("Helvetica-Bold", 10.0)
        c.drawString(2.0*cm, 1*
