import pandas as pd
import numpy as np

def pct(x):
    try:
        return f"{float(x):.2f}%"
    except:
        return "—"

def pts(x):
    try:
        return f"{float(x):.2f} pts"
    except:
        return "—"

def safe_int(x):
    try:
        return int(x)
    except:
        return 0

def reliability_takeaway(df: pd.DataFrame) -> dict:
    total = len(df)
    counts = df["Reliability"].value_counts(dropna=False).to_dict()

    high = safe_int(counts.get("High", 0))
    med = safe_int(counts.get("Medium", 0))
    directional = safe_int(counts.get("Directional", 0))
    low = safe_int(counts.get("Low", 0))

    clear = safe_int((df.get("Significant_95") == True).sum()) if "Significant_95" in df.columns else safe_int((df.get("P_Value", 1) <= 0.05).sum())

    headline = "Quick takeaway"
    if directional > (high + med):
        message = f"Most results are directional ({directional} of {total}). Treat these as signals to watch, not claims to publish."
    else:
        message = f"A meaningful share of the view is usable for decisions (High+Medium: {high+med} of {total})."

    sub = f"Statistically clear rows: {clear}. Directional rows: {directional}. Low confidence rows: {low}."
    return {"headline": headline, "message": message, "sub": sub}

def impact_quadrant_takeaway(df: pd.DataFrame) -> dict:
    # Expects columns: Quadrant, Lift_Pct, Diff_PctPts, Reliability, P_Value, Label
    if "Quadrant" not in df.columns:
        return {"headline": "Quick takeaway", "message": "Impact quadrants are not available in this view.", "sub": ""}

    q = df["Quadrant"].value_counts().to_dict()
    act = safe_int(q.get("Act", 0))
    watch = safe_int(q.get("Watch", 0))
    investigate = safe_int(q.get("Investigate", 0))
    ignore = safe_int(q.get("Ignore", 0))

    headline = "Quick takeaway"
    message = f"Act: {act} • Watch: {watch} • Investigate: {investigate} • Ignore: {ignore}."
    sub = "Use Act for messaging/investment decisions. Use Investigate to identify friction or negative signals worth fixing."
    return {"headline": headline, "message": message, "sub": sub}

def top_lists(df: pd.DataFrame, n=5):
    """
    Returns two tables:
    - top opportunities: Act quadrant, highest lift
    - top risks: Investigate quadrant, most negative lift
    """
    cols = [c for c in ["Label","Month Year","Brand","KPI","Diff_PctPts","Lift_Pct","P_Value","Reliability","Quadrant"] if c in df.columns]

    opp = df[df.get("Quadrant") == "Act"].copy() if "Quadrant" in df.columns else df.copy()
    opp = opp.sort_values("Lift_Pct", ascending=False).head(n) if "Lift_Pct" in opp.columns else opp.head(n)
    opp = opp[cols] if cols else opp

    risk = df[df.get("Quadrant") == "Investigate"].copy() if "Quadrant" in df.columns else df.copy()
    risk = risk.sort_values("Lift_Pct", ascending=True).head(n) if "Lift_Pct" in risk.columns else risk.head(n)
    risk = risk[cols] if cols else risk

    return opp, risk

def ci_reading_help():
    return {
        "title": "How to read the confidence interval ranking",
        "text": (
            "Each line is the most likely range for the true change. The dot is the best estimate.\n"
            "If the line crosses 0, the change may not be real. If it stays fully above/below 0, it’s stronger evidence."
        )
    }

def matrix_reading_help():
    return {
        "title": "How to read the impact matrix",
        "text": (
            "Dots: each dot is one KPI result.\n"
            "Lift (x-axis): right is positive, left is negative.\n"
            "Certainty (y-axis): higher means stronger evidence it’s real.\n"
            "Act: positive + clear. Watch: positive but not clear.\n"
            "Investigate: negative + clear. Ignore: negative but not clear."
        )
    }
