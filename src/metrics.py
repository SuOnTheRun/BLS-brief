import math
import pandas as pd


# -------------------------
# Helpers
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
# Core computation
# -------------------------

def compute_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes raw statistical outputs for BLS from inputs only.

    REQUIRED INPUT COLUMNS
    ----------------------
    - Control Sample
    - Exposed Sample
    - Control Score
    - Exposed Score

    Score format:
    - Either 0–100 (percent) OR 0–1 (proportion)
    - Auto-detected

    OUTPUT PHILOSOPHY
    -----------------
    This function:
    - Computes maths only
    - Does NOT assign reliability bands
    - Does NOT label clarity
    - Does NOT decide what is shown/hidden
    All interpretation happens downstream in rules.py
    """

    d = df.copy()

    required = [
        "Control Sample",
        "Exposed Sample",
        "Control Score",
        "Exposed Score"
    ]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")

    # -------------------------
    # Samples
    # -------------------------

    n_control = d["Control Sample"].apply(_to_float).astype(float)
    n_exposed = d["Exposed Sample"].apply(_to_float).astype(float)

    # -------------------------
    # Scores
    # -------------------------

    s_control = d["Control Score"].apply(_to_float).astype(float)
    s_exposed = d["Exposed Score"].apply(_to_float).astype(float)

    # Detect score scale
    med = pd.concat([s_control, s_exposed], axis=0).median(skipna=True)
    scores_are_percent = (med is not None) and (med == med) and (med > 1.5)

    if scores_are_percent:
        p_control = (s_control / 100.0).apply(_clamp01)
        p_exposed = (s_exposed / 100.0).apply(_clamp01)
    else:
        p_control = s_control.apply(_clamp01)
        p_exposed = s_exposed.apply(_clamp01)

    # -------------------------
    # Core effects
    # -------------------------

    diff_prop = p_exposed - p_control
    lift_prop = diff_prop / p_control.replace(0.0, float("nan"))

    # -------------------------
    # Two-proportion z-test
    # -------------------------

    pooled = ((p_control * n_control) + (p_exposed * n_exposed)) / (n_control + n_exposed)

    se = pooled * (1 - pooled) * (1 / n_control + 1 / n_exposed)
    se = se.apply(lambda x: math.sqrt(x) if (x == x and x >= 0) else float("nan"))

    z = diff_prop / se.replace(0.0, float("nan"))
    p_value = z.apply(lambda zz: _two_sided_p_value(zz) if (zz == zz) else float("nan"))

    # 95% confidence interval for the gap
    zcrit = 1.96
    ci_low = diff_prop - zcrit * se
    ci_high = diff_prop + zcrit * se

    # -------------------------
    # Effect size (Cohen’s h)
    # -------------------------

    def cohens_h(a, b):
        try:
            return (
                2.0 * math.asin(math.sqrt(_clamp01(b)))
                - 2.0 * math.asin(math.sqrt(_clamp01(a)))
            )
        except Exception:
            return float("nan")

    h_vals = [cohens_h(a, b) for a, b in zip(p_control.tolist(), p_exposed.tolist())]

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

    h_qual_vals = [h_qual(x) for x in h_vals]

    # -------------------------
    # Attach outputs (RAW ONLY)
    # -------------------------

    d["Control Sample"] = n_control
    d["Exposed Sample"] = n_exposed

    d["Control_Pct"] = p_control * 100.0
    d["Exposed_Pct"] = p_exposed * 100.0

    d["Diff_PctPts"] = diff_prop * 100.0
    d["Lift_Pct"] = lift_prop * 100.0

    d["Pooled_Prop"] = pooled
    d["Std_Error"] = se
    d["Z_Score"] = z
    d["P_Value"] = p_value

    d["CI_Low_PctPts"] = ci_low * 100.0
    d["CI_High_PctPts"] = ci_high * 100.0

    d["Effect_Size_h"] = h_vals
    d["Effect_Size_Qual"] = h_qual_vals

    # IMPORTANT:
    # No reliability band
    # No clarity label
    # No hiding logic
    # No notes
    # Those belong in rules.py

    return d
