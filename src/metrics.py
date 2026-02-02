import math
import pandas as pd


# =============================
# Small math helpers
# =============================
def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _two_sided_p_value(z: float) -> float:
    z = abs(float(z))
    return max(0.0, min(1.0, 2.0 * (1.0 - _norm_cdf(z))))


def _clamp01(x: float) -> float:
    if x != x:  # NaN
        return float("nan")
    return max(0.0, min(1.0, x))


def _to_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def _is_nan(x) -> bool:
    return not (x == x)


# =============================
# Notes dictionary (human-first)
# =============================
NOTES = {
    "LOW_RELIABILITY": "Only a small number of people answered this question. With small samples, results can swing easily—treat this as a hint, not a conclusion.",
    "DIRECTIONAL_SIGNAL": "This result points in a direction, but we don’t yet have enough evidence to be fully confident. Treat this as an early signal that would benefit from more data.",
    "UNCLEAR_SIGNAL": "We can’t reliably tell whether this difference is real or just random variation. This does not mean nothing happened—only that evidence is not strong yet.",
    "LIFT_HIDDEN_SMALL_BASE": "Relative change is hidden because the starting number is very small. When the baseline is tiny, lift can look huge and mislead. The gap out of 100 people is the clearest way to read this.",
    "FLAT_RESULT": "The exposed and control groups answered in very similar ways. This suggests the ads did not meaningfully change this metric in this period.",
    "NEGATIVE_RESULT": "Fewer people who saw the ads answered positively than those who didn’t. This can happen for many reasons and is a useful learning signal rather than a failure.",
}


# =============================
# Main function
# =============================
def compute_all_metrics(
    df: pd.DataFrame,
    # Movable bands (defaults)
    GREAT_THRESHOLD: int = 300,
    GOOD_THRESHOLD: int = 100,
    DIRECTIONAL_THRESHOLD: int = 50,
    # Clarity thresholds
    CLEAR_P_THRESHOLD: float = 0.05,
    DIRECTIONAL_P_THRESHOLD: float = 0.10,
    # Lift visibility rules
    MIN_BASELINE_PERCENT: float = 5.0,
    MIN_LIFT_SAMPLE: int = 50,
    # Flat threshold for notes
    FLAT_THRESHOLD_PP: float = 1.0,
) -> pd.DataFrame:
    """
    Computes BLS stats from inputs only.

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

    # Inputs
    n1 = d["Control Sample"].apply(_to_float).astype(float)
    n2 = d["Exposed Sample"].apply(_to_float).astype(float)

    s1 = d["Control Score"].apply(_to_float).astype(float)
    s2 = d["Exposed Score"].apply(_to_float).astype(float)

    # Decide if scores are % or proportions
    med = pd.concat([s1, s2], axis=0).median(skipna=True)
    scores_are_percent = (med is not None) and (med == med) and (med > 1.5)

    if scores_are_percent:
        p1 = (s1 / 100.0).apply(_clamp01)
        p2 = (s2 / 100.0).apply(_clamp01)
    else:
        p1 = s1.apply(_clamp01)
        p2 = s2.apply(_clamp01)

    # Core deltas
    diff = (p2 - p1)                              # proportion difference
    lift = diff / p1.replace(0.0, float("nan"))   # relative difference

    # Two-proportion z-test (pooled SE)
    pooled = ((p1 * n1) + (p2 * n2)) / (n1 + n2)

    se = (pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    se = se.apply(lambda x: math.sqrt(x) if (x == x and x >= 0) else float("nan"))

    z = diff / se.replace(0.0, float("nan"))
    pval = z.apply(lambda zz: _two_sided_p_value(zz) if (zz == zz) else float("nan"))

    # 95% CI for diff
    zcrit = 1.96
    ci_low = diff - zcrit * se
    ci_high = diff + zcrit * se

    sig95 = pval.apply(lambda pv: bool(pv <= CLEAR_P_THRESHOLD) if (pv == pv) else False)

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

    # =============================
    # Reliability + clarity (new model)
    # =============================
    reliability_n = pd.concat([n1, n2], axis=1).min(axis=1)

    def reliability_band(mn):
        if _is_nan(mn):
            return "Low"
        if mn >= GREAT_THRESHOLD:
            return "Great"
        if mn >= GOOD_THRESHOLD:
            return "Good"
        if mn >= DIRECTIONAL_THRESHOLD:
            return "Directional"
        return "Low"

    rel_band = reliability_n.apply(reliability_band)

    def clarity_band(pv, mn):
        if _is_nan(pv):
            return "Unclear"
        # "Clear" requires both strong p-value and at least Good sample size
        if pv <= CLEAR_P_THRESHOLD and (not _is_nan(mn)) and mn >= GOOD_THRESHOLD:
            return "Clear"
        if pv <= DIRECTIONAL_P_THRESHOLD:
            return "Directional"
        return "Unclear"

    cla_band = [clarity_band(pv, mn) for pv, mn in zip(pval.tolist(), reliability_n.tolist())]

    # =============================
    # Lift visibility + notes (explainers)
    # =============================
    control_pct = (p1 * 100.0)
    exposed_pct = (p2 * 100.0)
    diff_pp = (diff * 100.0)
    lift_pct = (lift * 100.0)

    lift_shown = []
    notes_short = []

    for cp, mn, pv, gap, rb, cb in zip(
        control_pct.tolist(),
        reliability_n.tolist(),
        pval.tolist(),
        diff_pp.tolist(),
        rel_band.tolist(),
        cla_band,
    ):
        notes = []

        # Lift visibility
        show_lift = True
        if (not _is_nan(cp) and cp < MIN_BASELINE_PERCENT) or (not _is_nan(mn) and mn < MIN_LIFT_SAMPLE):
            show_lift = False
            notes.append("LIFT_HIDDEN_SMALL_BASE")

        # Reliability / clarity notes
        if rb == "Low":
            notes.append("LOW_RELIABILITY")
        if cb == "Directional":
            notes.append("DIRECTIONAL_SIGNAL")
        if cb == "Unclear":
            notes.append("UNCLEAR_SIGNAL")

        # Flat / negative notes
        if not _is_nan(gap):
            if abs(gap) < FLAT_THRESHOLD_PP:
                notes.append("FLAT_RESULT")
            if gap < 0:
                notes.append("NEGATIVE_RESULT")

        # Deduplicate while keeping order
        seen = set()
        notes = [x for x in notes if not (x in seen or seen.add(x))]

        lift_shown.append(bool(show_lift))

        if len(notes) == 0:
            notes_short.append("")
        else:
            # Keep it short: join up to 2 notes
            msg = " ".join([NOTES[k] for k in notes[:2] if k in NOTES])
            notes_short.append(msg)

    # =============================
    # Write outputs (keeps your old columns too)
    # =============================
    d["Control_Pct"] = control_pct
    d["Exposed_Pct"] = exposed_pct

    d["Diff_PctPts"] = diff_pp
    d["Lift_Pct"] = lift_pct

    d["Pooled_Prop"] = pooled
    d["Std_Error"] = se
    d["Z_Score"] = z
    d["P_Value"] = pval

    d["CI_Low_PctPts"] = (ci_low * 100.0)
    d["CI_High_PctPts"] = (ci_high * 100.0)

    d["Significant_95"] = sig95
    d["Effect_Size_h"] = h
    d["Effect_Size_Qual"] = hq

    # New fields (used by charts + UI + PDF if you want)
    d["Reliability_N"] = reliability_n
    d["Reliability_Band"] = rel_band
    d["Clarity_Band"] = cla_band
    d["Lift_Shown"] = lift_shown
    d["Notes_Short"] = notes_short

    return d
