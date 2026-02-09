import math
import pandas as pd


# -----------------------------
# Notes copy dictionary (human)
# -----------------------------
NOTES_COPY = {
    "LOW_RELIABILITY": "Only a small number of people answered this question. Treat this as a hint rather than a conclusion.",
    "DIRECTIONAL_SIGNAL": "This points in a direction, but we don’t yet have enough evidence to be fully confident. It would benefit from more responses.",
    "UNCLEAR_SIGNAL": "We can’t reliably tell whether this difference is real or random variation yet.",
    "LIFT_HIDDEN_SMALL_BASE": "Relative change is hidden because the starting number is very small. The difference out of 100 people is the clearest way to read this.",
    "FLAT_RESULT": "The exposed and control groups answered in very similar ways. This suggests the ads did not meaningfully change this metric during the measured period.",
    "NEGATIVE_RESULT": "Fewer people who saw the ads answered positively compared to those who did not. This is a useful learning signal (message fit, context, timing).",
    "AGGREGATED_RESULT": "This value is an average across multiple periods or results. Individual waves may perform better or worse than this overall number.",
    "MIXED_PERFORMANCE": "This average combines stronger and weaker results. The overall number shows direction, but underlying performance is mixed.",
    "DATA_MISSING": "Some results are not shown because there were not enough responses to interpret them reliably.",
    "FILTER_IMPACT": "The current filters significantly narrow the data shown. Results may differ with a broader time period or more markets.",
}


def _notes_from_keys(keys: str):
    if not keys:
        return []
    parts = [k.strip() for k in str(keys).split("|") if k.strip()]
    return [NOTES_COPY.get(k, k) for k in parts]


def _safe_float(x, default=float("nan")):
    try:
        return float(x)
    except Exception:
        return default


def _pp_to_people(pp):
    """Convert pp to 'out of 100 people' language."""
    if pp != pp:
        return ""
    if pp > 0:
        return f"About {pp:.0f} more out of 100 people"
    if pp < 0:
        return f"About {abs(pp):.0f} fewer out of 100 people"
    return "About the same out of 100 people"


def _kpi_verb(kpi_name: str):
    """Deterministic deck-style verbs by KPI name (editable)."""
    k = (kpi_name or "").lower()

    if any(x in k for x in ["awareness", "unaided", "aided", "recall", "remember"]):
        return ("memorability", "improving brand recall")
    if any(x in k for x in ["consider", "consideration"]):
        return ("consideration", "shifting consideration")
    if any(x in k for x in ["intent", "purchase", "buy", "shop"]):
        return ("purchase momentum", "influencing intent")
    if any(x in k for x in ["favor", "favour", "lik", "preference"]):
        return ("brand equity", "strengthening preference")
    if any(x in k for x in ["visit", "store", "footfall", "traffic"]):
        return ("behavioral conversion", "driving visitation")
    return ("brand response", "moving the metric")


def _strength_word(clarity: str, reliability: str):
    if clarity == "Clear" and reliability in ("Great", "Good"):
        return "strong, decision-safe"
    if clarity == "Clear" and reliability == "Directional":
        return "real, but fragile"
    if clarity == "Directional":
        return "promising, not confirmed"
    return "uncertain"


def _recommendation(quadrant: str, clarity: str, reliability: str, gap_pp: float):
    """
    Deterministic "so what" logic.
    """
    if quadrant == "Act" and clarity == "Clear" and reliability in ("Great", "Good"):
        return "Scale this learning: keep creative cues consistent and extend reach/flight."
    if quadrant == "Act" and clarity != "Clear":
        return "Treat as a leading indicator: keep running, but confirm with more responses before over-claiming."
    if quadrant == "Investigate":
        return "Investigate friction: check creative/message fit, placement context, and audience mismatch."
    if quadrant == "Watch" and gap_pp >= 0:
        return "Watch and validate: this may be the start of a real shift—prioritise more sample and repeat measurement."
    return "Deprioritise for now: evidence does not support action yet."


def _infer_quadrant(row):
    """
    Use your existing matrix logic (effect vs certainty) if present.
    If not present, infer a simple one:
      - positive & clear-ish => Act/Watch
      - negative & clear-ish => Investigate
      - else Ignore
    """
    gap = _safe_float(row.get("Diff_PctPts"))
    clarity = str(row.get("Clarity_Band", "Unclear"))
    if gap != gap:
        return "Ignore"
    if gap >= 0 and clarity == "Clear":
        return "Act"
    if gap >= 0 and clarity != "Clear":
        return "Watch"
    if gap < 0 and clarity in ("Clear", "Directional"):
        return "Investigate"
    return "Ignore"


def write_row_insight(row, style="deck"):
    """
    Returns dict:
      headline, evidence, so_what, notes(list)
    """
    brand = str(row.get("Brand", "")).strip()
    kpi = str(row.get("KPI", "")).strip()
    month = str(row.get("Month Year", "")).strip()

    control = _safe_float(row.get("Control_Pct"))
    exposed = _safe_float(row.get("Exposed_Pct"))
    gap_pp = _safe_float(row.get("Diff_PctPts"))
    lift = _safe_float(row.get("Lift_Pct"))
    p = _safe_float(row.get("P_Value"))
    n1 = _safe_float(row.get("Control Sample"))
    n2 = _safe_float(row.get("Exposed Sample"))

    reliability = str(row.get("Reliability_Band", "Low"))
    clarity = str(row.get("Clarity_Band", "Unclear"))
    lift_shown = bool(row.get("Lift_Shown", True))

    quadrant = str(row.get("Quadrant", "")) or _infer_quadrant(row)

    noun, verb_phrase = _kpi_verb(kpi)
    strength = _strength_word(clarity, reliability)

    direction = "up" if (gap_pp == gap_pp and gap_pp > 0) else "down" if (gap_pp == gap_pp and gap_pp < 0) else "flat"
    people_line = _pp_to_people(gap_pp)

    # Headline (deck voice)
    if direction == "up":
        headline = f"{brand}: {verb_phrase} on {kpi}."
    elif direction == "down":
        headline = f"{brand}: pressure on {kpi} (needs investigation)."
    else:
        headline = f"{brand}: {kpi} is stable (no meaningful movement)."

    # Evidence line
    # Keep it crisp, like a slide footnote.
    lift_part = f" • Lift: {lift:.1f}%" if (lift_shown and lift == lift) else ""
    p_part = f" • p={p:.4f}" if p == p else ""
    n_part = ""
    if n1 == n1 and n2 == n2:
        n_part = f" • n={int(n1)} vs {int(n2)}"

    evidence = (
        f"{people_line} said yes after seeing the ads "
        f"(Control {control:.1f}% vs Exposed {exposed:.1f}%, Gap {gap_pp:.2f}pp{lift_part})"
        f"{p_part}{n_part}. "
        f"Signal strength: {strength} ({reliability} reliability, {clarity} clarity)."
    )

    so_what = _recommendation(quadrant, clarity, reliability, gap_pp)

    notes = _notes_from_keys(row.get("Notes_Keys", ""))

    return {
        "headline": headline,
        "evidence": evidence,
        "so_what": so_what,
        "notes": notes,
        "quadrant": quadrant,
    }


def add_insights_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - Insight_Headline
      - Insight_Evidence
      - Insight_SoWhat
      - Insight_Notes (joined)
      - Insight_Quadrant
    """
    d = df.copy()
    out = [write_row_insight(r) for _, r in d.iterrows()]
    d["Insight_Headline"] = [x["headline"] for x in out]
    d["Insight_Evidence"] = [x["evidence"] for x in out]
    d["Insight_SoWhat"] = [x["so_what"] for x in out]
    d["Insight_Notes"] = ["\n".join(x["notes"]) if x["notes"] else "" for x in out]
    d["Insight_Quadrant"] = [x["quadrant"] for x in out]
    return d


def write_view_insights(view_df: pd.DataFrame, top_n: int = 5):
    """
    Generates a simple "deck summary" from the current filtered view.
    """
    if view_df is None or len(view_df) == 0:
        return {
            "headline": "No results in view.",
            "wins": [],
            "risks": [],
            "watch": [],
        }

    d = view_df.copy()

    # Prioritise: Clear+Good/Great first, then biggest absolute gap
    def score_row(r):
        gap = _safe_float(r.get("Diff_PctPts"), 0.0)
        rel = str(r.get("Reliability_Band", "Low"))
        cla = str(r.get("Clarity_Band", "Unclear"))
        bonus = 0.0
        if cla == "Clear":
            bonus += 100.0
        if rel in ("Great", "Good"):
            bonus += 50.0
        return bonus + abs(gap)

    d["_priority"] = [score_row(r) for _, r in d.iterrows()]

    # Wins: positive + Act-ish
    wins = d[d["Diff_PctPts"].astype(float) > 0].sort_values("_priority", ascending=False).head(top_n)
    risks = d[d["Diff_PctPts"].astype(float) < 0].sort_values("_priority", ascending=False).head(top_n)

    # Watch: Directional clarity but positive movement
    watch = d[(d.get("Clarity_Band", "") == "Directional") & (d["Diff_PctPts"].astype(float) > 0)].sort_values("_priority", ascending=False).head(top_n)

    wins_txt = [write_row_insight(r)["headline"] for _, r in wins.iterrows()]
    risks_txt = [write_row_insight(r)["headline"] for _, r in risks.iterrows()]
    watch_txt = [write_row_insight(r)["headline"] for _, r in watch.iterrows()]

    headline = f"In this view: {len(view_df)} results. Prioritise the clearest, most reliable movements first."

    return {
        "headline": headline,
        "wins": wins_txt,
        "risks": risks_txt,
        "watch": watch_txt,
    }
