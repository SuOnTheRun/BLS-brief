import math
import pandas as pd


# -----------------------
# parsing + math helpers
# -----------------------
def _to_float(x):
    try:
        if x is None:
            return float("nan")
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if s == "" or s.lower() in ("nan", "none", "null", "-"):
            return float("nan")
        s = s.replace(",", "")
        if s.endswith("%"):
            s = s[:-1].strip()
        return float(s)
    except Exception:
        return float("nan")


def _clamp01(x: float) -> float:
    if x != x:
        return float("nan")
    return max(0.0, min(1.0, x))


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _two_sided_p_value(z: float) -> float:
    z = abs(float(z))
    return max(0.0, min(1.0, 2.0 * (1.0 - _norm_cdf(z))))


# -----------------------
# main compute
# -----------------------
def compute_all_metrics(
    df: pd.DataFrame,
    *,
    # Reliability bands (movable)
    GREAT_THRESHOLD: int = 300,
    GOOD_THRESHOLD: int = 100,
    DIRECTIONAL_THRESHOLD: int = 50,
    # Clarity thresholds (p-value)
    CLEAR_P_THRESHOLD: float = 0.05,
    DIRECTIONAL_P_THRESHOLD: float = 0.10,
    # Lift visibility
    MIN_BASELINE_PERCENT: float = 5.0,
    MIN_LIFT_SAMPLE: int = 50,
    # Flat threshold
    FLAT_THRESHOLD_PP: float = 1.0,
) -> pd.DataFrame:
    """
    INPUT columns required (from your template):
      - Control Sample
      - Exposed Sample
      - Control Score
      - Exposed Score

    Control/Exposed Score can be:
      - 0–100 (percent)
      - 0–1 (proportion)
      - "38%" string

    OUTPUT adds:
      Control_Pct, Exposed_Pct
      Diff_PctPts (gap)
      Lift_Pct
      P_Value, CI_Low_PctPts, CI_High_PctPts, Significant_95
      Reliability_N, Reliability_Band (Great/Good/Directional/Low)
      Clarity_Band (Clear/Directional/Unclear)
      Lift_Shown
      Notes_Keys (pipe-separated)
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

    # Decide if scores are % or proportions:
    # If median > 1.5 we treat as percent scale.
    med = pd.concat([s1, s2], axis=0).median(skipna=True)
    scores_are_percent = (med == med) and (med > 1.5)

    if scores_are_percent:
        p1 = (s1 / 100.0).apply(_clamp01)
        p2 = (s2 / 100.0).apply(_clamp01)
    else:
        p1 = s1.apply(_clamp01)
        p2 = s2.apply(_clamp01)

    # Gap + Lift (in proportions first)
    diff = (p2 - p1)  # 0..1
    lift = diff / p1.replace(0.0, float("nan"))

    # Two-proportion z test (pooled)
    pooled = ((p1 * n1) + (p2 * n2)) / (n1 + n2)
    se = pooled * (1 - pooled) * (1 / n1 + 1 / n2)
    se = se.apply(lambda x: math.sqrt(x) if (x == x and x >= 0) else float("nan"))

    z = diff / se.replace(0.0, float("nan"))
    pval = z.apply(lambda zz: _two_sided_p_value(zz) if (zz == zz) else float("nan"))

    # 95% CI for diff
    zcrit = 1.96
    ci_low = diff - zcrit * se
    ci_high = diff + zcrit * se

    # Derived display columns
    control_pct = p1 * 100.0
    exposed_pct = p2 * 100.0
    diff_pp = diff * 100.0
    lift_pct = lift * 100.0

    # Reliability
    reliability_n = pd.concat([n1, n2], axis=1).min(axis=1)

    def _reliability_band(mn):
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

    rel_band = reliability_n.apply(_reliability_band)

    # Clarity (gated by "Good" sample for Clear)
    def _clarity_band(pv, mn):
        try:
            pv = float(pv)
            mn = float(mn)
        except Exception:
            return "Unclear"
        if pv != pv:
            return "Unclear"
        if pv <= CLEAR_P_THRESHOLD and (mn == mn and mn >= GOOD_THRESHOLD):
            return "Clear"
        if pv <= DIRECTIONAL_P_THRESHOLD:
            return "Directional"
        return "Unclear"

    clarity_band = [
        _clarity_band(pv, mn) for pv, mn in zip(pval.tolist(), reliability_n.tolist())
    ]

    # Lift shown rule
    def _lift_shown(cpct, cn):
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

    lift_shown = [
        _lift_shown(cp, cn) for cp, cn in zip(control_pct.tolist(), n1.tolist())
    ]

    # Notes keys
    def _notes(gappp, rb, cb, lift_ok):
        notes = []
        if rb == "Low":
            notes.append("LOW_RELIABILITY")
        if cb == "Unclear":
            notes.append("UNCLEAR_SIGNAL")
        elif cb == "Directional":
            notes.append("DIRECTIONAL_SIGNAL")
        if not lift_ok:
            notes.append("LIFT_HIDDEN_SMALL_BASE")

        try:
            g = float(gappp)
            if g == g:
                if abs(g) < float(FLAT_THRESHOLD_PP):
                    notes.append("FLAT_RESULT")
                elif g < 0:
                    notes.append("NEGATIVE_RESULT")
        except Exception:
            pass

        # stable ordering, de-dup
        seen = set()
        out = []
        for k in notes:
            if k not in seen:
                seen.add(k)
                out.append(k)
        return out

    notes_keys = [
        "|".join(_notes(g, rb, cb, lk))
        for g, rb, cb, lk in zip(diff_pp.tolist(), rel_band.tolist(), clarity_band, lift_shown)
    ]

    # Significant_95
    sig95 = [bool(pv <= CLEAR_P_THRESHOLD) if (pv == pv) else False for pv in pval.tolist()]

    # Write columns
    d["Control_Pct"] = control_pct
    d["Exposed_Pct"] = exposed_pct
    d["Diff_PctPts"] = diff_pp
    d["Lift_Pct"] = lift_pct

    d["P_Value"] = pval
    d["CI_Low_PctPts"] = ci_low * 100.0
    d["CI_High_PctPts"] = ci_high * 100.0
    d["Significant_95"] = sig95

    d["Reliability_N"] = reliability_n
    d["Reliability_Band"] = rel_band
    d["Clarity_Band"] = clarity_band
    d["Lift_Shown"] = lift_shown
    d["Notes_Keys"] = notes_keys

    return d
