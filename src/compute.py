import math
import pandas as pd


# -----------------------------
# Small math helpers (no scipy needed)
# -----------------------------
def _to_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def _norm_cdf(z: float) -> float:
    # Standard normal CDF using erf (built-in)
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _two_sided_p_value(z: float) -> float:
    # two-sided p-value
    cdf = _norm_cdf(abs(z))
    return max(0.0, min(1.0, 2.0 * (1.0 - cdf)))


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _safe_div(a: float, b: float):
    if b == 0 or (isinstance(b, float) and math.isnan(b)):
        return float("nan")
    return a / b


# -----------------------------
# Public API expected by app.py
# -----------------------------
def compute_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all columns from samples + scores.

    Expected input columns (case-sensitive):
      - Control Sample
      - Exposed Sample
      - Control Score
      - Exposed Score

    Scores can be in percent (0-100) or proportion (0-1). We auto-detect.

    Returns df with added computed fields used across the app and PDF.
    """
    d = df.copy()

    required = ["Control Sample", "Exposed Sample", "Control Score", "Exposed Score"]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")

    # Cast samples
    d["Control Sample"] = d["Control Sample"].apply(_to_float)
    d["Exposed Sample"] = d["Exposed Sample"].apply(_to_float)

    # Cast scores
    d["Control Score"] = d["Control Score"].apply(_to_float)
    d["Exposed Score"] = d["Exposed Score"].apply(_to_float)

    # Detect score scale (percent vs proportion)
    # If median > 1.5 assume percent.
    med = pd.concat([d["Control Score"], d["Exposed Score"]], axis=0).median(skipna=True)
    scores_are_percent = (med is not None) and (not math.isnan(med)) and (med > 1.5)

    if scores_are_percent:
        control_p = (d["Control Score"] / 100.0).astype(float)
        exposed_p = (d["Exposed Score"] / 100.0).astype(float)
    else:
        control_p = d["Control Score"].astype(float)
        exposed_p = d["Exposed Score"].astype(float)

    # Clamp to [0,1] to avoid nonsense from dirty sheets
    control_p = control_p.apply(_clamp01)
    exposed_p = exposed_p.apply(_clamp01)

    n1 = d["Control Sample"].astype(float)
    n2 = d["Exposed Sample"].astype(float)

    # Basic metrics
    diff = (exposed_p - control_p)               # proportion difference
    lift = diff / control_p.replace(0, float("nan"))

    # Pooled proportion for z-test
    pooled = ((control_p * n1) + (exposed_p * n2)) / (n1 + n2)

    # Standard error for difference in proportions
    se = (pooled * (1 - pooled) * (1 / n1 + 1 / n2)).apply(
        lambda x: math.sqrt(x) if (x is not None and not math.isnan(x) and x >= 0) else float("nan")
    )

    z = diff / se.replace(0, float("nan"))
    pval = z.apply(lambda zz: _two_sided_p_value(zz) if (zz is not None and not math.isnan(zz)) else float("nan"))

    # Confidence interval around diff (95%)
    zcrit = 1.96
    ci_low = diff - zcrit * se
    ci_high = diff + zcrit * se

    # Significance
    sig95 = pval.apply(lambda pv: bool(pv <= 0.05) if (pv is not None and not math.isnan(pv)) else False)

    # Cohen's h (effect size for proportions)
    # h = 2*arcsin(sqrt(p2)) - 2*arcsin(sqrt(p1))
    def cohens_h(p1, p2):
        try:
            return 2.0 * math.asin(math.sqrt(_clamp01(p2))) - 2.0 * math.asin(math.sqrt(_clamp01(p1)))
        except Exception:
            return float("nan")

    h = [cohens_h(a, b) for a, b in zip(control_p.tolist(), exposed_p.tolist())]

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

    # Reliability label (plain + practical)
    # You can tune thresholds later; this gives a clean start.
    def reliability(nc, ne, pv):
        try:
            nc = float(nc)
            ne = float(ne)
            pv = float(pv)
        except Exception:
            return "Low"

        if math.isnan(nc) or math.isnan(ne):
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

    rel = [
        reliability(a, b, c)
        for a, b, c in zip(n1.tolist(), n2.tolist(), pval.tolist())
    ]

    # Data flags for transparency
    flags = []
    for a, b, cs, es in zip(n1.tolist(), n2.tolist(), d["Control Score"].tolist(), d["Exposed Score"].tolist()):
        f = []
        if (a is None) or (b is None) or math.isnan(a) or math.isnan(b) or a <= 0 or b <= 0:
            f.append("Bad samples")
        if (cs is None) or (es is None) or math.isnan(cs) or math.isnan(es):
            f.append("Bad scores")
        flags.append(", ".join(f) if f else "")

    # Write computed columns (both "pretty" and machine-friendly names)
    d["Control_Pct"] = (control_p * 100.0)
    d["Exposed_Pct"] = (exposed_p * 100.0)

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

    d["Data_Flag"] = flags

    return d
