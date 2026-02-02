import math
import pandas as pd


# -----------------------------
# math helpers
# -----------------------------
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


# -----------------------------
# Notes dictionary (keys → copy)
# -----------------------------
NOTES_COPY = {
    "LOW_RELIABILITY": "Only a small number of people answered this question. With small samples, results can swing easily, so treat this as a hint rather than a conclusion.",
    "DIRECTIONAL_SIGNAL": "This points in a clear direction, but we don’t yet have enough evidence to be fully confident. This often happens when the number of responses is moderate rather than large.",
    "UNCLEAR_SIGNAL": "We can’t reliably tell whether this difference is real or just random variation. This does not mean nothing happened — it means we don’t have enough evidence yet.",
    "LIFT_HIDDEN_SMALL_BASE": "Relative change is hidden because the starting number is very small. When the baseline is tiny, percentages can look much bigger than they really are and can be misleading. The difference out of 100 people is the clearest way to understand this result.",
    "FLAT_RESULT": "The exposed and control groups answered in very similar ways. This suggests the ads did not meaningfully change this metric during the measured period.",
    "NEGATIVE_RESULT": "Fewer people who saw the ads answered positively compared to those who did not. This can happen for many reasons and is a useful learning signal rather than a failure.",
    "AGGREGATED_RESULT": "This value is an average across multiple periods or results. Individual waves may perform better or worse than this overall number.",
    "MIXED_PERFORMANCE": "This average combines both stronger and weaker results. The overall number shows direction, but underlying performance is mixed.",
    "DATA_MISSING": "Some results are not shown because there were not enough responses to interpret them reliably.",
    "FILTER_IMPACT": "The current filters significantly narrow the data shown. Results may differ when viewing a broader time period or additional markets.",
}

NOTES_PRIORITY = [
    "DATA_MISSING",
    "LOW_RELIABILITY",
    "UNCLEAR_SIGNAL",
    "DIRECTIONAL_SIGNAL",
    "LIFT_HIDDEN_SMALL_BASE",
    "FILTER_IMPACT",
    "AGGREGATED_RESULT",
    "MIXED_PERFORMANCE",
    "NEGATIVE_RESULT",
    "FLAT_RESULT",
]


def _notes_to_text(keys):
    keys = [k for k in keys if k in NOTES_COPY]
    # dedupe while preserving order
    seen = set()
    out = []
    for k in keys:
        if k not in seen:
            out.append(k)
            seen.add(k)
    # sort by priority
    out = sorted(out, key=lambda k: NOTES_PRIORITY.index(k) if k in NOTES_PRIORITY else 999)
    short = "; ".join([NOTES_COPY[k] for k in out[:2]])  # keep it compact
    return out, short


# -----------------------------
# Scoring / bands
# -----------------------------
def _reliability_band(min_n: float, great=300, good=100, directional=50):
    """
    Movable bands based on min(control_n, exposed_n).
    Returns: Great/Good/Directional/Low
    """
    try:
        mn = float(min_n)
    except Exception:
        return "Low"
    if mn >= great:
        return "Great"
    if mn >= good:
        return "Good"
    if mn >= directional:
        return "Directional"
    return "Low"


def _clarity_band(p_value: float, min_n: float, good_threshold=100, clear_p=0.05, directional_p=0.10):
    """
    Clear only if p<=clear_p AND min_n>=good_threshold.
    Directional if p<=directional_p.
    Else Unclear.
    """
    try:
        pv = float(p_value)
        mn = float(min_n)
    except Exception:
        return "Unclear"

    if pv == pv and pv <= clear_p and mn >= good_threshold:
        return "Clear"
    if pv == pv and pv <= directional_p:
        return "Directional"
    return "Unclear"


def _legacy_reliability_label(rel_band: str):
    """
    Keeps your existing chart labels stable:
    Great -> High
    Good -> Medium
    Directional -> Directional
    Low -> Low
    """
    mapping = {"Great": "High", "Good": "Medium", "Directional": "Directional", "Low": "Low"}
    return mapping.get(rel_band, "Low")


# -----------------------------
# Main compute
# -----------------------------
def compute_all_metrics(
    df: pd.DataFrame,
    *,
    # Movable bands (defaults match your spec)
    GREAT_THRESHOLD: int = 300,
    GOOD_THRESHOLD: int = 100,
    DIRECTIONAL_THRESHOLD: int = 50,
    # Clarity thresholds
    CLEAR_P_THRESHOLD: float = 0.05,
    DIRECTIONAL_P_THRESHOLD: float = 0.10,
    # Lift hiding rules
    MIN_BASELINE_PERCENT: float = 5.0,
    MIN_LIFT_SAMPLE: int = 50,
    # Flat threshold
    FLAT_THRESHOLD_PP: float = 1.0,
) -> pd.DataFrame:
    """
    Inputs-only computation.

    Required input columns:
      - Control Sample
      - Exposed Sample
      - Control Score
      - Exposed Score

    Scores can be 0–100 (percent) or 0–1 (proportion).

    Adds:
      Control_Pct, Exposed_Pct
      Diff_PctPts (gap in pp), Lift_Pct (relative)
      P_Value, CI_Low_PctPts, CI_High_PctPts
      Reliability_N, Reliability_Band, Clarity_Band
      Reliability (legacy label: High/Medium/Directional/Low)
      Lift_Shown (bool)
      Notes_Keys (list as string), Notes_Short (string)
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("compute_all_metrics expects a pandas DataFrame.")

    d = df.copy()

    required = ["Control Sample", "Exposed Sample", "Control Score", "Exposed Score"]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")

    n1 = d["Control Sample"].apply(_to_float).astype(float)
    n2 = d["Exposed Sample"].apply(_to_float).astype(float)

    s1 = d["Control Score"].apply(_to_float).astype(float)
    s2 = d["Exposed Score"].apply(_to_float).astype(float)

    # Decide if scores look like percents
    med = pd.concat([s1, s2], axis=0).median(skipna=True)
    scores_are_percent = (med is not None) and (med == med) and (med > 1.5)

    if scores_are_percent:
        p1 = (s1 / 100.0).apply(_clamp01)
        p2 = (s2 / 100.0).apply(_clamp01)
    else:
        p1 = s1.apply(_clamp01)
        p2 = s2.apply(_clamp01)

    # Core deltas
    diff = (p2 - p1)                       # proportion
    gap_pp = diff * 100.0                  # percentage points
    lift = diff / p1.replace(0.0, float("nan"))
    lift_pct = lift * 100.0

    pooled = ((p1 * n1) + (p2 * n2)) / (n1 + n2)

    se = (pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    se = se.apply(lambda x: math.sqrt(x) if (x == x and x >= 0) else float("nan"))

    z = diff / se.replace(0.0, float("nan"))
    pval = z.apply(lambda zz: _two_sided_p_value(zz) if (zz == zz) else float("nan"))

    # 95% CI on the gap
    zcrit = 1.96
    ci_low = diff - zcrit * se
    ci_high = diff + zcrit * se

    # Reliability + clarity + lift visibility + notes
    reliability_n = pd.concat([n1, n2], axis=1).min(axis=1)

    rel_band = [
        _reliability_band(mn, great=GREAT_THRESHOLD, good=GOOD_THRESHOLD, directional=DIRECTIONAL_THRESHOLD)
        for mn in reliability_n.tolist()
    ]
    clarity = [
        _clarity_band(pv, mn, good_threshold=GOOD_THRESHOLD, clear_p=CLEAR_P_THRESHOLD, directional_p=DIRECTIONAL_P_THRESHOLD)
        for pv, mn in zip(pval.tolist(), reliability_n.tolist())
    ]
    rel_legacy = [_legacy_reliability_label(rb) for rb in rel_band]

    # Lift hiding rule
    control_pct = p1 * 100.0
    lift_shown = ~(
        (control_pct < float(MIN_BASELINE_PERCENT)) |
        (n1 < float(MIN_LIFT_SAMPLE))
    )

    # Notes engine (row-level)
    notes_keys = []
    for mn, cb, rb, gp, ls, cp in zip(reliability_n.tolist(), clarity, rel_band, gap_pp.tolist(), lift_shown.tolist(), control_pct.tolist()):
        keys = []
        if rb == "Low":
            keys.append("LOW_RELIABILITY")
        if cb == "Directional":
            keys.append("DIRECTIONAL_SIGNAL")
        if cb == "Unclear":
            keys.append("UNCLEAR_SIGNAL")
        if not bool(ls):
            keys.append("LIFT_HIDDEN_SMALL_BASE")
        try:
            if float(gp) < 0:
                keys.append("NEGATIVE_RESULT")
            if abs(float(gp)) < float(FLAT_THRESHOLD_PP):
                keys.append("FLAT_RESULT")
        except Exception:
            pass
        keys, short = _notes_to_text(keys)
        notes_keys.append((keys, short))

    d["Control_Pct"] = (p1 * 100.0)
    d["Exposed_Pct"] = (p2 * 100.0)

    d["Diff_PctPts"] = gap_pp
    d["Lift_Pct"] = lift_pct

    d["Pooled_Prop"] = pooled
    d["Std_Error"] = se
    d["Z_Score"] = z
    d["P_Value"] = pval

    d["CI_Low_PctPts"] = (ci_low * 100.0)
    d["CI_High_PctPts"] = (ci_high * 100.0)

    d["Reliability_N"] = reliabili_]()
