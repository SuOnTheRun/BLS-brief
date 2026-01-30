import math
import pandas as pd


# -------------------------
# Stats helpers
# -------------------------
def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _two_sided_p_value(z: float) -> float:
    z = abs(float(z))
    return max(0.0, min(1.0, 2.0 * (1.0 - _norm_cdf(z))))


def _clamp01(x: float) -> float:
    if x != x:
        return float("nan")
    return max(0.0, min(1.0, x))


def _to_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


# -------------------------
# Notes dictionary (human-first)
# -------------------------
NOTES = {
    "LOW_RELIABILITY": "Only a small number of people answered this question. Treat this as a hint, not a conclusion.",
    "DIRECTIONAL_SIGNAL": "This points in a direction, but we don’t have enough evidence to be fully confident yet. Treat as an early signal.",
    "UNCLEAR_SIGNAL": "We can’t reliably tell if this difference is real or random variation yet.",
    "LIFT_HIDDEN_SMALL_BASE": "Relative change is hidden because the baseline is very small. The difference out of 100 people is the clearest way to read this result.",
    "FLAT_RESULT": "Exposed and control answered very similarly. This suggests the ads didn’t meaningfully change this metric in this period.",
    "NEGATIVE_RESULT": "Fewer people who saw the ads answered positively than those who didn’t. This can be a learning signal (message fit, timing, context).",
    "AGGREGATED_RESULT": "This value is an average across multiple results. Individual rows can be higher or lower than the overall number.",
    "MIXED_PERFORMANCE": "This combines both stronger and weaker results. The overall direction is useful, but underlying performance is mixed.",
    "DATA_MISSING": "Some results are not shown because there were not enough responses to interpret them reliably.",
    "FILTER_IMPACT": "Current filters significantly narrow the data. Results may differ when viewing a broader time period or more markets.",
}


def _join_notes(note_keys):
    """Return a short, readable line from multiple notes (keeps it compact)."""
    keys = [k for k in note_keys if k in NOTES]
    if not keys:
        return ""
    # show max 2 in short form
    short = [NOTES[k] for k in keys[:2]]
    if len(keys) > 2:
        short.append("More context available in Notes.")
    return " ".join(short)


# -------------------------
# Core computation
# -------------------------
def compute_all_metrics(
    df: pd.DataFrame,
    *,
    GREAT_THRESHOLD: int = 300,
    GOOD_THRESHOLD: int = 100,
    DIRECTIONAL_THRESHOLD: int = 50,
    CLEAR_P_THRESHOLD: float = 0.05,
    DIRECTIONAL_P_THRESHOLD: float = 0.10,
    MIN_BASELINE_PERCENT: float = 5.0,
    MIN_LIFT_SAMPLE: int = 50,
    FLAT_THRESHOLD_PTS: float = 1.0,
) -> pd.DataFrame:
    """
    Computes the full BLS stats from inputs only + a simple interpretation layer.

    Required input columns:
      - Control Sample
      - Exposed Sample
      - Control Score
      - Exposed Score

    Scores can be 0–100 (percent) or 0–1 (proportion).
    """
    d = df.copy()

    required = ["Control Sample", "Exposed Sample", "Control Score", "Exposed Score"]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")

    n1 = d["Control Sample"].apply(_to_float).astype(float)
    n2 = d["Exposed Sample"].apply(_to_float).astype(float)

    s1 = d["Control Score"].apply(_to_float).astype(float)
    s2 = d["Exposed Score"].apply(_to_float).astype(float)

    # detect percent vs proportion
    med = pd.concat([s1, s2], axis=0).median(skipna=True)
    scores_are_percent = (med is not None) and (med == med) and (med > 1.5)

    if scores_are_percent:
        p1 = (s1 / 100.0).apply(_clamp01)
        p2 = (s2 / 100.0).apply(_clamp01)
    else:
        p1 = s1.apply(_clamp01)
        p2 = s2.apply(_clamp01)

    diff = (p2 - p1)  # proportion
    pooled = ((p1 * n1) + (p2 * n2)) / (n1 + n2)

    se = (pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    se = se.apply(lambda x: math.sqrt(x) if (x == x and x >= 0) else float("nan"))

    z = diff / se.replace(0.0, float("nan"))
    pval = z.apply(lambda zz: _two_sided_p_value(zz) if (zz == zz) else float("nan"))

    zcrit = 1.96
    ci_low = diff - zcrit * se
    ci_high = diff + zcrit * se

    # Effect size (Cohen's h)
    def cohens_h(a, b):
        try:
            return 2.0 * math.asin(math.sqrt(_clamp01(b))) - 2.0 * math.asin(math.sqrt(_clamp01(a)))
        except Exception:
            return float("nan")

    h = [cohens_h(a, b) for a, b in zip(p1.tolist(), p2.tolist())]

    def h_qual(val):
        try:
            av = abs(float(val))
            if av < 0.2:
                return "Small"
            if av < 0.5:
                return "Medium"
            return "Large"
        except Exception:
            return "Unknown"

    hq = [h_qual(x) for x in h]

    # -------------------------
    # Derived + display metrics
    # -------------------------
    control_pct = (p1 * 100.0)
    exposed_pct = (p2 * 100.0)
    gap_pp = (diff * 100.0)

    # lift is relative: (exposed-control)/control
    lift_raw = diff / p1.replace(0.0, float("nan"))
    lift_pct = (lift_raw * 100.0)

    reliability_n = pd.concat([n1, n2], axis=1).min(axis=1)

    # Reliability bands (movable)
    def reliability_band(min_n):
        try:
            mn = float(min_n)
        except Exception:
            return "Low"
        if mn >= GREAT_THRESHOLD:
            return "Great"
        if mn >= GOOD_THRESHOLD:
            return "Good"
        if mn >= DIRECTIONAL_THRESHOLD:
            return "Directional"
        return "Low"

    rel_band = [reliability_band(x) for x in reliability_n.tolist()]

    # Statistical clarity bands
    def clarity_band(pv, min_n):
        try:
            pv = float(pv)
            mn = float(min_n)
        except Exception:
            return "Unclear"
        # Clear requires both p-value and at least "Good" sample
        if (pv <= CLEAR_P_THRESHOLD) and (mn >= GOOD_THRESHOLD):
            return "Clear"
        if pv <= DIRECTIONAL_P_THRESHOLD:
            return "Directional"
        return "Unclear"

    clr_band = [clarity_band(pv, mn) for pv, mn in zip(pval.tolist(), reliability_n.tolist())]

    # Lift visibility rule
    lift_hidden = []
    for c_pct, c_n in zip(control_pct.tolist(), n1.tolist()):
        try:
            if float(c_pct) < float(MIN_BASELINE_PERCENT) or float(c_n) < float(MIN_LIFT_SAMPLE):
                lift_hidden.append(True)
            else:
                lift_hidden.append(False)
        except Exception:
            lift_hidden.append(True)

    lift_pct_visible = lift_pct.copy()
    # hide lift by setting NaN (UI can show "—" + notes)
    lift_pct_visible = lift_pct_visible.where(pd.Series(lift_hidden).map(lambda x: not x), other=float("nan"))

    # -------------------------
    # Notes engine
    # -------------------------
    notes_keys = []
    notes_short = []
    notes_full = []

    for gp, lpct, pv, rb, cb, cn, mn in zip(
        gap_pp.tolist(),
        lift_pct_visible.tolist(),
        pval.tolist(),
        rel_band,
        clr_band,
        n1.tolist(),
        reliability_n.tolist(),
    ):
        keys = []

        # reliability note
        if rb == "Low":
            keys.append("LOW_RELIABILITY")

        # clarity notes
        if cb == "Directional":
            keys.append("DIRECTIONAL_SIGNAL")
        if cb == "Unclear":
            keys.append("UNCLEAR_SIGNAL")

        # lift hidden note
        if lpct != lpct:  # NaN means hidden
            keys.append("LIFT_HIDDEN_SMALL_BASE")

        # flat / negative
        try:
            if abs(float(gp)) < float(FLAT_THRESHOLD_PTS):
                keys.append("FLAT_RESULT")
            if float(gp) < 0:
                keys.append("NEGATIVE_RESULT")
        except Exception:
            pass

        # de-dup, keep order
        seen = set()
        keys = [k for k in keys if not (k in seen or seen.add(k))]

        notes_keys.append(keys)
        notes_short.append(_join_notes(keys))
        notes_full.append("\n".join([f"- {NOTES[k]}" for k in keys]) if keys else "")

    # -------------------------
    # Human-language sentence layer
    # -------------------------
    def _out_of_100_sentence(ctrl_pct, exp_pct, gap_pp):
        try:
            g = float(gap_pp)
            if g > 0:
                return f"Out of 100 people, about {abs(g):.0f} more said yes after seeing the ads."
            if g < 0:
                return f"Out of 100 people, about {abs(g):.0f} fewer said yes after seeing the ads."
            return "Out of 100 people, the two groups answered almost the same."
        except Exception:
            return ""

    def _confidence_sentence(rb, cb):
        if rb in ("Great", "Good") and cb == "Clear":
            return "We can be confident this change is real (enough responses + strong signal)."
        if cb == "Directional":
            return "This looks like an early signal, but we’d want more responses to be fully confident."
        if cb == "Unclear":
            return "This could be random variation — treat as a hint unless it repeats."
        return "Treat with caution."

    human_what = [
        _out_of_100_sentence(c, e, g)
        for c, e, g in zip(control_pct.tolist(), exposed_pct.tolist(), gap_pp.tolist())
    ]
    human_conf = [_confidence_sentence(rb, cb) for rb, cb in zip(rel_band, clr_band)]

    # -------------------------
    # Output columns
    # -------------------------
    d["Control_Pct"] = control_pct
    d["Exposed_Pct"] = exposed_pct
    d["Diff_PctPts"] = gap_pp

    # both: raw lift + visible lift
    d["Lift_Pct_Raw"] = lift_pct
    d["Lift_Pct"] = lift_pct_visible  # may be NaN when hidden
    d["Lift_Hidden"] = pd.Series(lift_hidden, index=d.index)

    d["Pooled_Prop"] = pooled
    d["Std_Error"] = se
    d["Z_Score"] = z
    d["P_Value"] = pval
    d["CI_Low_PctPts"] = (ci_low * 100.0)
    d["CI_High_PctPts"] = (ci_high * 100.0)

    d["Significant_95"] = d["P_Value"].apply(lambda pv: bool(pv <= 0.05) if (pv == pv) else False)

    d["Effect_Size_h"] = h
    d["Effect_Size_Qual"] = hq

    # new governance columns
    d["Reliability_n"] = reliability_n
    d["Reliability_Band"] = rel_band
    d["Clarity_Band"] = clr_band
    d["Notes_Keys"] = notes_keys
    d["Notes_Short"] = notes_short
    d["Notes_Full"] = notes_full

    # legacy compatibility (so old charts/app code won’t break)
    # Map Great/Good/Directional/Low -> High/Medium/Directional/Low for older visuals.
    def legacy_reliability(rb):
        if rb == "Great":
            return "High"
        if rb == "Good":
            return "Medium"
        if rb == "Directional":
            return "Directional"
        return "Low"

    d["Reliability"] = [legacy_reliability(x) for x in rel_band]

    # human narrative columns
    d["Human_WhatHappened"] = human_what
    d["Human_Confidence"] = human_conf

    return d
