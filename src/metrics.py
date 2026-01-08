import numpy as np
import pandas as pd
from scipy.stats import norm
from .config import DEFAULT_THRESHOLDS

def _to_prop_from_score(x):
    """
    Accepts:
      - "47.10%" -> 0.471
      - 47.10    -> 0.471 (assumed percent if > 1.5)
      - 0.471    -> 0.471
    """
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        s = x.strip().replace(",", "")
        if s.endswith("%"):
            s = s[:-1]
        try:
            v = float(s)
        except:
            return np.nan
    else:
        try:
            v = float(x)
        except:
            return np.nan

    if v > 1.5:
        return v / 100.0
    return v

def _effect_size_h(p1, p2):
    # Cohen's h for proportions
    p1 = np.clip(p1, 1e-12, 1 - 1e-12)
    p2 = np.clip(p2, 1e-12, 1 - 1e-12)
    return 2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1))

def _effect_size_qual(h):
    ah = np.abs(h)
    if np.isnan(ah):
        return ""
    if ah < 0.2:
        return "Small"
    if ah < 0.5:
        return "Medium"
    return "Large"

def _reliability(significant, n1, n2, data_flag, effect_qual):
    """
    Plain, practical confidence label:
    - High: clear + healthy samples
    - Medium: clear but limited sample
    - Directional: not clear, but enough sample to treat as a signal (not a conclusion)
    - Low: too small or too noisy
    """
    if significant and data_flag == "":
        return "High"
    if significant and data_flag in ["Limited sample", "Low sample"]:
        return "Medium"
    if (not significant) and data_flag == "" and effect_qual in ["Medium", "Large"]:
        return "Directional"
    if data_flag == "Low sample":
        return "Low"
    return "Directional" if not significant else "Medium"

def compute_metrics(df: pd.DataFrame, alpha: float = DEFAULT_THRESHOLDS.alpha) -> pd.DataFrame:
    out = df.copy()

    out["Control Sample"] = pd.to_numeric(out["Control Sample"], errors="coerce")
    out["Exposed Sample"] = pd.to_numeric(out["Exposed Sample"], errors="coerce")

    out["Control_Prop"] = out["Control Score"].apply(_to_prop_from_score)
    out["Exposed_Prop"] = out["Exposed Score"].apply(_to_prop_from_score)

    p1 = out["Control_Prop"].astype(float)
    p2 = out["Exposed_Prop"].astype(float)
    n1 = out["Control Sample"].astype(float)
    n2 = out["Exposed Sample"].astype(float)

    # Basics
    out["Diff_Prop"] = p2 - p1
    out["Lift_Rel"] = np.where(p1 == 0, np.nan, (p2 - p1) / p1)

    out["Control_Pct"] = p1 * 100
    out["Exposed_Pct"] = p2 * 100
    out["Diff_PctPts"] = out["Diff_Prop"] * 100
    out["Lift_Pct"] = out["Lift_Rel"] * 100

    # Compatibility with your sheet naming
    out["Uplift_Average"] = out["Lift_Pct"]
    out["Uplift_Median"] = np.nan  # cannot compute without individual-level data

    # Totals
    out["Total_Sample"] = n1 + n2

    # Two-proportion z-test (sheet-style)
    pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    out["Pooled_Prop"] = pooled

    std_error = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    out["Std_Error"] = std_error

    z = (p2 - p1) / std_error
    out["Z_Score"] = z

    pval = 2 * (1 - norm.cdf(np.abs(z)))
    out["P_Value"] = pval
    out["Significant_95"] = out["P_Value"] < alpha

    # SE of the difference (unpooled)
    se_diff = np.sqrt((p1 * (1 - p1)) / n1 + (p2 * (1 - p2)) / n2)
    out["SE_Diff"] = se_diff

    zcrit = norm.ppf(1 - alpha / 2)
    ci_low = (p2 - p1) - zcrit * se_diff
    ci_high = (p2 - p1) + zcrit * se_diff

    out["CI_Diff_Low"] = ci_low
    out["CI_Diff_High"] = ci_high
    out["CI_Low_PctPts"] = ci_low * 100
    out["CI_High_PctPts"] = ci_high * 100

    # Effect size
    h = _effect_size_h(p1, p2)
    out["Effect_Size_h"] = h
    out["Effect_Size_Qual"] = [ _effect_size_qual(v) for v in h ]

    # Data flags (simple)
    out["Data_Flag"] = ""
    out.loc[(n1 < DEFAULT_THRESHOLDS.min_n_low) | (n2 < DEFAULT_THRESHOLDS.min_n_low), "Data_Flag"] = "Low sample"
    out.loc[
        ((n1 >= DEFAULT_THRESHOLDS.min_n_low) & (n1 < DEFAULT_THRESHOLDS.min_n_warn)) |
        ((n2 >= DEFAULT_THRESHOLDS.min_n_low) & (n2 < DEFAULT_THRESHOLDS.min_n_warn)),
        "Data_Flag"
    ] = out["Data_Flag"].replace("", "Limited sample")

    # Reliability
    out["Reliability"] = [
        _reliability(bool(sig), float(a), float(b), str(flag), str(eq))
        for sig, a, b, flag, eq in zip(out["Significant_95"], n1, n2, out["Data_Flag"], out["Effect_Size_Qual"])
    ]

    return out
