import math
import pandas as pd

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

def compute_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the full BLS stats from inputs only.

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

    med = pd.concat([s1, s2], axis=0).median(skipna=True)
    scores_are_percent = (med is not None) and (med == med) and (med > 1.5)

    if scores_are_percent:
        p1 = (s1 / 100.0).apply(_clamp01)
        p2 = (s2 / 100.0).apply(_clamp01)
    else:
        p1 = s1.apply(_clamp01)
        p2 = s2.apply(_clamp01)

    diff = (p2 - p1)
    lift = diff / p1.replace(0.0, float("nan"))

    pooled = ((p1 * n1) + (p2 * n2)) / (n1 + n2)

    se = (pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    se = se.apply(lambda x: math.sqrt(x) if (x == x and x >= 0) else float("nan"))

    z = diff / se.replace(0.0, float("nan"))
    pval = z.apply(lambda zz: _two_sided_p_value(zz) if (zz == zz) else float("nan"))

    zcrit = 1.96
    ci_low = diff - zcrit * se
    ci_high = diff + zcrit * se

    sig95 = pval.apply(lambda pv: bool(pv <= 0.05) if (pv == pv) else False)

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

    def reliability(nc, ne, pv):
        try:
            nc = float(nc); ne = float(ne); pv = float(pv)
        except Exception:
            return "Low"

        min_n = min(nc, ne)

        if pv <= 0.05 and min_n >= 300:
            return "High"
        if pv <= 0.05 and min_n >= 150:
            return "Medium"
        if pv <= 0.10 and min_n >= 150:
            return "Directional"
        if min_n >= 150 and pv > 0.10:
            return "Directional"
        return "Low"

    rel = [reliability(a, b, c) for a, b, c in zip(n1.tolist(), n2.tolist(), pval.tolist())]

    d["Control_Pct"] = (p1 * 100.0)
    d["Exposed_Pct"] = (p2 * 100.0)

    d["Diff_PctPts"] = (diff * 100.0)
    d["Lift_Pct"] = (lift * 100.0)

    d["Pooled_Prop"] = pooled
    d["Std_Error"] = se
    d["Z_Score"] = z
    d["P_Value"] = pval

    d["CI_Low_PctPts"] = (ci_low * 100.0)
    d["CI_High_PctPts"] = (ci_high * 100.0)

    d["Significant_95"] = sig95
    d["Effect_Size_h"] = h
    d["Effect_Size_Qual"] = hq
    d["Reliability"] = rel

    return d
