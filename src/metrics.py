import math
import pandas as pd


# =========================
# Math helpers
# =========================
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


# =========================
# Notes dictionary (keys only here; copy lives in UI layer too)
# =========================
NOTE_KEYS = {
    "LOW_RELIABILITY",
    "DIRECTIONAL_SIGNAL",
    "UNCLEAR_SIGNAL",
    "LIFT_HIDDEN_SMALL_BASE",
    "FLAT_RESULT",
    "NEGATIVE_RESULT",
    "AGGREGATED_RESULT",
    "MIXED_PERFORMANCE",
    "DATA_MISSING",
    "FILTER_IMPACT",
}


def compute_all_metrics(
    df: pd.DataFrame,
    *,
    # Reliability bands (movable)
    GREAT_THRESHOLD: int = 300,
    GOOD_THRESHOLD: int = 100,
    DIRECTIONAL_THRESHOLD: int = 50,
    # Clarity thresholds
    CLEAR_P_THRESHOLD: float = 0.05,
    DIRECTIONAL_P_THRESHOLD: float = 0.10,
    # Lift visibility rules
    MIN_BASELINE_PERCENT: float = 5.0,  # control % < 5 => hide lift
    MIN_LIFT_SAMPLE: int = 50,          # control_n < 50 => hide lift
    # Flat result rule (gap smaller than this)
    FLAT_THRESHOLD_PP: float = 1.0,
) -> pd.DataFrame:
    """
    Computes BLS stats from inputs only, plus:
      - Reliability_N, Reliability_Band (Great/Good/Directional/Low)
      - Clarity_Band (Clear/Directional/Unclear)
      - Lift_Shown boolean
      - Notes_Keys (pipe-separated), Notes_Count

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

    # detect whether scores are 0-100 vs 0-1
    med = pd.concat([s1, s2], axis=0).median(skipna=True)
    scores_are_percent = (med is not None) and (med == med) and (med > 1.5)

    if scores_are_percent:
        p1 = (s1 / 100.0).apply(_clamp01)
        p2 = (s2 / 100.0).apply(_clamp01)
    else:
        p1 = s1.apply(_clamp01)
        p2 = s2.apply(_clamp01)

    diff = (p2 - p1)  # proportion points (0-1)
    lift = diff / p1.replace(0.0, float("nan"))

    pooled = ((p1 * n1) + (p2 * n2)) / (n1 + n2)

    se = (pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    se = se.apply(lambda x: math.sqrt(x) if (x == x and x >= 0) else float("nan"))

    z = diff / se.replace(0.0, float("nan"))
    pval = z.apply(lambda zz: _two_sided_p_value(zz) if (zz == zz) else float("nan"))

    zcrit = 1.96
    ci_low = diff - zcrit * se
    ci_high = diff + zcrit * se

    sig95 = pval.apply(lambda pv: bool(pv <= CLEAR_P_THRESHOLD) if (pv == pv) else False)

    # effect size (Cohen's h)
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
    # Reliability + Clarity bands (A)
    # -------------------------
    reliability_n = pd.concat([n1, n2], axis=1).min(axis=1)

    def reliability_band(mn):
        try:
            mn = float(mn)
        except Exception:
            return "Low"
        if mn != mn:
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
        try:
            pv = float(pv)
            mn = float(mn)
        except Exception:
            return "Unclear"
        if pv != pv:
            return "Unclear"
        # Gate Clear by "Good" minimum sample
        if pv <= CLEAR_P_THRESHOLD and (mn == mn and mn >= GOOD_THRESHOLD):
            return "Clear"
        if pv <= DIRECTIONAL_P_THRESHOLD:
            return "Directional"
        return "Unclear"

    clarity = [clarity_band(pv, mn) for pv, mn in zip(pval.tolist(), reliability_n.tolist())]

    # -------------------------
    # Lift visibility rule (A)
    # -------------------------
    control_pct = (p1 * 100.0)
    exposed_pct = (p2 * 100.0)
    diff_pp = (diff * 100.0)
    lift_pct = (lift * 100.0)

    def lift_shown_row(cpct, cn):
        try:
            cpct = float(cpct)
            cn = float(cn)
        except Exception:
            return False
        if cpct != cpct or cn != cn:
            return False
        if cpct < MIN_BASELINE_PERCENT:
            return False
        if cn < MIN_LIFT_SAMPLE:
            return False
        return True

    lift_shown = [lift_shown_row(cp, cn) for cp, cn in zip(control_pct.tolist(), n1.tolist())]

    # -------------------------
    # Notes engine (keys)
    # -------------------------
    def notes_for_row(gap_pp_val, rb, cb, lift_ok):
        notes = []
        # reliability
        if rb == "Low":
            notes.append("LOW_RELIABILITY")
        # clarity
        if cb == "Unclear":
            notes.append("UNCLEAR_SIGNAL")
        elif cb == "Directional":
            notes.append("DIRECTIONAL_SIGNAL")
        # lift hidden
        if not lift_ok:
            notes.append("LIFT_HIDDEN_SMALL_BASE")
        # direction / flatness
        try:
            g = float(gap_pp_val)
            if g == g:
                if abs(g) < float(FLAT_THRESHOLD_PP):
                    notes.append("FLAT_RESULT")
                elif g < 0:
                    notes.append("NEGATIVE_RESULT")
        except Exception:
            pass

        # de-dup preserve order
        seen = set()
        out = []
        for k in notes:
            if k in NOTE_KEYS and k not in seen:
                seen.add(k)
                out.append(k)
        return out

    notes_list = [
        notes_for_row(g, rb, cb, lk)
        for g, rb, cb, lk in zip(diff_pp.tolist(), rel_band.tolist(), clarity, lift_shown)
    ]
    notes_keys = ["|".join(x) if x else "" for x in notes_list]
    notes_count = [len(x) for x in notes_list]

    # -------------------------
    # Write columns
    # -------------------------
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

    # A-system fields
    d["Reliability_N"] = reliability_n
    d["Reliability_Band"] = rel_band
    d["Clarity_Band"] = clarity
    d["Lift_Shown"] = lift_shown

    # Notes (keys)
    d["Notes_Keys"] = notes_keys
    d["Notes_Count"] = notes_count

    return d
