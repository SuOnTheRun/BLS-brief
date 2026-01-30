import io
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

from .charts import (
    chart_control_vs_exposed_matplotlib,
    chart_lift_rank_matplotlib,
    chart_confidence_quadrant_matplotlib,
    fig_to_png_bytes,
)


# -----------------------------
# Layout helpers
# -----------------------------
def _header(c, title, subtitle):
    c.setFont("Helvetica-Bold", 15)
    c.drawString(2.0 * cm, 28.4 * cm, str(title)[:90])

    c.setFont("Helvetica", 9.5)
    c.drawString(2.0 * cm, 27.8 * cm, str(subtitle)[:110])

    c.setLineWidth(0.6)
    c.line(2.0 * cm, 27.5 * cm, 19.5 * cm, 27.5 * cm)


def _stat_card(c, x, y, w, h, label, value, note):
    c.setLineWidth(0.6)
    c.roundRect(x, y, w, h, 8, stroke=1, fill=0)

    c.setFont("Helvetica", 8.8)
    c.drawString(x + 0.5 * cm, y + h - 0.7 * cm, str(label)[:40])

    c.setFont("Helvetica-Bold", 14)
    c.drawString(x + 0.5 * cm, y + h - 1.65 * cm, str(value)[:18])

    c.setFont("Helvetica", 8.4)
    c.drawString(x + 0.5 * cm, y + 0.45 * cm, str(note)[:58])


def _wrap_lines(text, max_chars=110):
    if not text:
        return []
    s = str(text).replace("\r", " ").replace("\n", " ").strip()
    if not s:
        return []
    out = []
    while len(s) > max_chars:
        cut = s.rfind(" ", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        out.append(s[:cut].strip())
        s = s[cut:].strip()
    if s:
        out.append(s)
    return out


def _safe_float(x, default=0.0):
    try:
        v = float(x)
        return v
    except Exception:
        return default


def _band_rank(val, mapping):
    return mapping.get(str(val), 0)


# -----------------------------
# Narrative helpers (plain language)
# -----------------------------
def _direction_word(gap_pp):
    if gap_pp > 0:
        return "more"
    if gap_pp < 0:
        return "fewer"
    return "about the same"


def _recommendation_strength(rel_band, cla_band):
    rel_band = str(rel_band)
    cla_band = str(cla_band)
    if cla_band == "Clear" and rel_band in ("Good", "Great"):
        return "Safe to cite as evidence and act on."
    if cla_band == "Directional" and rel_band in ("Good", "Great"):
        return "Promising signal. Worth monitoring or testing again before making a hard claim."
    if cla_band == "Clear" and rel_band in ("Directional", "Low"):
        return "Looks real, but sample size is not ideal. Use cautiously."
    return "Treat as a hint. Don’t use as a hard conclusion yet."


def _short_context_line(row):
    bits = []
    for k in ["Month Year", "Market", "Category"]:
        v = row.get(k, "")
        if v is not None and str(v).strip():
            bits.append(str(v).strip())
    return " • ".join(bits)


# -----------------------------
# Public API (matches app.py)
# -----------------------------
def build_pdf_bytes(df, title="BLS Brief", include_comparisons=True, max_deep_dives=12):
    """
    Build a PDF report from a dataframe.

    Expected columns (best effort; will degrade gracefully):
      - Brand, KPI, Month Year, Market, Category
      - Control Sample, Exposed Sample
      - Control_Pct, Exposed_Pct
      - Diff_PctPts, Lift_Pct, Lift_Visible
      - P_Value, CI_Low_PctPts, CI_High_PctPts
      - Reliability_Band, Clarity_Band
      - Notes_Short, Notes_Full
    """
    if df is None or len(df) == 0:
        # Return a simple "empty report" PDF
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        dt = datetime.now().strftime("%d %b %Y")
        _header(c, title, f"Generated {dt}")
        c.setFont("Helvetica", 10.0)
        c.drawString(2.0 * cm, 25.5 * cm, "No rows available for export.")
        c.save()
        buf.seek(0)
        return buf.getvalue()

    d = df.copy()

    # Some callers pass a single row Series in a df-like way; ensure df
    if not hasattr(d, "iterrows"):
        d = d.to_frame().T

    # Core safe fields
    if "Diff_PctPts" not in d.columns:
        d["Diff_PctPts"] = 0.0
    if "Lift_Pct" not in d.columns:
        d["Lift_Pct"] = float("nan")
    if "Lift_Visible" not in d.columns:
        d["Lift_Visible"] = True
    if "Reliability_Band" not in d.columns:
        d["Reliability_Band"] = ""
    if "Clarity_Band" not in d.columns:
        d["Clarity_Band"] = ""
    if "Notes_Short" not in d.columns:
        d["Notes_Short"] = ""
    if "Notes_Full" not in d.columns:
        d["Notes_Full"] = ""

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)

    dt = datetime.now().strftime("%d %b %Y")
    subtitle = f"Generated {dt}"

    # -----------------------------
    # Page 1 — Summary
    # -----------------------------
    _header(c, title, subtitle)

    total = int(len(d))
    clear_count = int((d["Clarity_Band"].astype(str) == "Clear").sum()) if "Clarity_Band" in d.columns else 0
    avg_gap = float(d["Diff_PctPts"].astype(float).mean()) if total else 0.0

    # Lift average (only where visible)
    if "Lift_Visible" in d.columns:
        lift_visible_rows = d[d["Lift_Visible"].astype(bool)]
        avg_lift = float(lift_visible_rows["Lift_Pct"].astype(float).mean()) if len(lift_visible_rows) else float("nan")
    else:
        avg_lift = float(d["Lift_Pct"].astype(float).mean()) if total else float("nan")

    _stat_card(c, 2.0 * cm, 24.9 * cm, 4.6 * cm, 2.1 * cm, "Rows in view", f"{total}", "Exported results")
    _stat_card(c, 6.9 * cm, 24.9 * cm, 4.6 * cm, 2.1 * cm, "Clear results", f"{clear_count}", "Clarity = Clear")
    _stat_card(c, 11.8 * cm, 24.9 * cm, 4.6 * cm, 2.1 * cm, "Average gap", f"{avg_gap:.2f} pp", "Exposed − Control")

    lift_note = "Across lift-visible rows" if (avg_lift == avg_lift) else "Lift hidden where baseline small"
    lift_val = f"{avg_lift:.2f}%" if (avg_lift == avg_lift) else "—"
    _stat_card(c, 16.7 * cm, 24.9 * cm, 2.8 * cm, 2.1 * cm, "Avg lift", lift_val, lift_note[:18])

    # Headline highlights (4 bullets)
    c.setFont("Helvetica-Bold", 10.5)
    c.drawString(2.0 * cm, 24.0 * cm, "Highlights")

    tmp = d.copy()
    tmp["AbsGap"] = tmp["Diff_PctPts"].astype(float).abs()

    # Biggest win/risk by gap
    biggest_win = tmp.sort_values("Diff_PctPts", ascending=False).iloc[0]
    biggest_risk = tmp.sort_values("Diff_PctPts", ascending=True).iloc[0]

    # Most reliable (prefer Clear + Good/Great + high Reliability_N if present)
    rel_map = {"Great": 4, "Good": 3, "Directional": 2, "Low": 1}
    cla_map = {"Clear": 3, "Directional": 2, "Unclear": 1}

    tmp["RelRank"] = tmp["Reliability_Band"].astype(str).map(rel_map).fillna(0).astype(int)
    tmp["ClaRank"] = tmp["Clarity_Band"].astype(str).map(cla_map).fillna(0).astype(int)
    if "Reliability_N" in tmp.columns:
        tmp["Reliability_N"] = tmp["Reliability_N"].astype(float)
    else:
        tmp["Reliability_N"] = 0.0

    most_reliable = tmp.sort_values(["ClaRank", "RelRank", "Reliability_N", "AbsGap"], ascending=False).iloc[0]
    needs_caution = tmp.sort_values(["ClaRank", "RelRank", "Reliability_N"], ascending=True).iloc[0]

    bullets = [
        ("Biggest win", biggest_win),
        ("Biggest risk", biggest_risk),
        ("Most reliable", most_reliable),
        ("Needs caution", needs_caution),
    ]

    y = 23.4 * cm
    c.setFont("Helvetica", 9.4)
    for label, r in bullets:
        brand = str(r.get("Brand", "")).strip()
        kpi = str(r.get("KPI", "")).strip()
        gap = _safe_float(r.get("Diff_PctPts", 0.0))
        relb = str(r.get("Reliability_Band", ""))
        clab = str(r.get("Clarity_Band", ""))
        direction = _direction_word(gap)
        line = f"{label}: {brand} • {kpi} — {abs(gap):.1f} {direction} out of 100 (Reliability {relb}, Clarity {clab})"
        c.drawString(2.0 * cm, y, line[:120])
        y -= 0.55 * cm

    # Comparisons charts (if enough rows)
    if include_comparisons and total > 1:
        # Lift rank chart: only lift-visible rows if possible, else use gap as a fallback proxy
        rank_df = d.copy()
        if "Lift_Visible" in rank_df.columns:
            r2 = rank_df[rank_df["Lift_Visible"].astype(bool)].copy()
            rank_df = r2 if len(r2) else rank_df

        if "Label" not in rank_df.columns:
            rank_df["Label"] = rank_df.get("Brand", "").astype(str) + " • " + rank_df.get("KPI", "").astype(str)

        # If Lift_Pct is mostly missing, use gap in that chart so it's still informative
        lift_nan_ratio = float(rank_df["Lift_Pct"].isna().mean()) if "Lift_Pct" in rank_df.columns else 1.0
        if lift_nan_ratio > 0.6:
            rank_df["Lift_Pct"] = rank_df.get("Diff_PctPts", 0.0)

        fig1 = chart_lift_rank_matplotlib(rank_df[["Label", "Lift_Pct"]], title="Ranked effects (higher is better)")
        img1 = ImageReader(io.BytesIO(fig_to_png_bytes(fig1)))
        c.drawImage(img1, 2.0 * cm, 13.1 * cm, width=17.5 * cm, height=7.6 * cm, preserveAspectRatio=True, mask="auto")

        fig2 = chart_confidence_quadrant_matplotlib(d, title="Effect vs certainty")
        img2 = ImageReader(io.BytesIO(fig_to_png_bytes(fig2)))
        c.drawImage(img2, 2.0 * cm, 5.4 * cm, width=17.5 * cm, height=7.0 * cm, preserveAspectRatio=True, mask="auto")

        # Tiny reading cue (simple, no jargon)
        c.setFont("Helvetica", 8.6)
        c.drawString(2.0 * cm, 4.8 * cm, "Reading tip: prioritize big effects that also look stable (higher certainty and stronger samples).")

    c.showPage()

    # -----------------------------
    # Deep dives (caps at max_deep_dives)
    # Sort by absolute gap desc, then clarity/reliability
    # -----------------------------
    dd = d.copy()
    dd["AbsGap"] = dd["Diff_PctPts"].astype(float).abs()
    dd["RelRank"] = dd["Reliability_Band"].astype(str).map(rel_map).fillna(0).astype(int)
    dd["ClaRank"] = dd["Clarity_Band"].astype(str).map(cla_map).fillna(0).astype(int)
    dd = dd.sort_values(["ClaRank", "RelRank", "AbsGap"], ascending=False)

    max_pages = int(min(max_deep_dives, len(dd)))
    for i in range(max_pages):
        row = dd.iloc[i]

        brand = str(row.get("Brand", "")).strip()
        kpi = str(row.get("KPI", "")).strip()
        ctx = _short_context_line(row)

        gap = _safe_float(row.get("Diff_PctPts", 0.0))
        lift = row.get("Lift_Pct", float("nan"))
        lift_visible = bool(row.get("Lift_Visible", True))

        relb = str(row.get("Reliability_Band", ""))
        clab = str(row.get("Clarity_Band", ""))
        pval = _safe_float(row.get("P_Value", float("nan")))

        control_pct = _safe_float(row.get("Control_Pct", float("nan")))
        exposed_pct = _safe_float(row.get("Exposed_Pct", float("nan")))

        title_line = f"{brand} — {kpi}"
        subtitle_line = ctx if ctx else subtitle
        _header(c, title_line, subtitle_line)

        # Chart
        fig = chart_control_vs_exposed_matplotlib(row)
        img = ImageReader(io.BytesIO(fig_to_png_bytes(fig)))
        c.drawImage(img, 2.0 * cm, 17.2 * cm, width=17.5 * cm, height=8.6 * cm, preserveAspectRatio=True, mask="auto")

        # Interpretation block
        c.setFont("Helvetica-Bold", 10.0)
        c.drawString(2.0 * cm, 16.2 * cm, "Summary")

        direction_word = _direction_word(gap)
        sentence = f"Out of 100 people, about {abs(gap):.1f} {direction_word} said yes after seeing the ads."
        c.setFont("Helvetica", 9.4)
        c.drawString(2.0 * cm, 15.7 * cm, sentence[:120])

        # Confidence line (human)
        conf_line = f"Reliability: {relb} • Clarity: {clab}"
        if pval == pval:
            conf_line += f" • p={pval:.4f}"
        c.setFont("Helvetica", 9.2)
        c.drawString(2.0 * cm, 15.2 * cm, conf_line[:120])

        # What to do
        c.setFont("Helvetica-Bold", 10.0)
        c.drawString(2.0 * cm, 14.4 * cm, "So what")
        c.setFont("Helvetica", 9.4)
        rec = _recommendation_strength(relb, clab)
        for j, ln in enumerate(_wrap_lines(rec, max_chars=110)[:3]):
            c.drawString(2.0 * cm, (13.9 - 0.5 * j) * cm, ln[:120])

        # Numbers (keep clean)
        y0 = 12.0 * cm
        c.setFont("Helvetica-Bold", 9.6)
        c.drawString(2.0 * cm, y0, "Numbers")
        c.setFont("Helvetica", 9.2)
        c.drawString(2.0 * cm, (y0 - 0.55 * cm), f"Control: {control_pct:.1f}% (n={int(_safe_float(row.get('Control Sample',0)))} )")
        c.drawString(2.0 * cm, (y0 - 1.10 * cm), f"Exposed: {exposed_pct:.1f}% (n={int(_safe_float(row.get('Exposed Sample',0)))} )")
        c.drawString(2.0 * cm, (y0 - 1.65 * cm), f"Gap: {gap:.2f} points (exposed minus control)")

        if lift_visible and (lift == lift):
            c.drawString(2.0 * cm, (y0 - 2.20 * cm), f"Lift: {float(lift):.1f}%")
        else:
            c.drawString(2.0 * cm, (y0 - 2.20 * cm), "Lift: hidden (baseline/sample too small)")

        # Notes (only if present)
        notes_full = str(row.get("Notes_Full", "")).strip()
        notes_short = str(row.get("Notes_Short", "")).strip()
        if notes_full or notes_short:
            c.setFont("Helvetica-Bold", 9.6)
            c.drawString(2.0 * cm, 8.8 * cm, "Notes")
            c.setFont("Helvetica", 9.0)
            text = notes_full if notes_full else notes_short
            lines = _wrap_lines(text, max_chars=110)[:6]
            y = 8.3 * cm
            for ln in lines:
                c.drawString(2.0 * cm, y, ln[:120])
                y -= 0.48 * cm

        c.showPage()

    c.save()
    buf.seek(0)
    return buf.getvalue()
