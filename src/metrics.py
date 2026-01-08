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

def compute_metrics(df: pd.DataFrame, alpha: float = DEFAULT_THRESHOLDS.alpha) -> pd.DataFrame:
    out = df.copy()

    # numeric
    out["Control Sample"] = pd.to_numeric(out["Control Sample"], errors="coerce")
    out["Exposed Sample"] = pd.to_numeric(out["Exposed Sample"], errors="coerce")

    # derive proportions strictly from scores (input surface)
    out["Control_Prop"] = out["Control Score"].apply(_to_prop_from_score)
    out["Exposed_Prop"] = out["Exposed Score"].apply(_to_prop_from_score)

    p1 = out["Control_Prop"].astype(float)
    p2 = out["Exposed_Prop"].astype(float)
    n1 = out["Control Sample"].astype(float)
    n2 = out["Exposed Sample"].astype(float)

    # diffs
    out["Diff_Prop"] = p2 - p1
    out["Lift_Rel"] = np.where(p1 == 0, np.nan, (p2 - p1) / p1)

    # two-proportion z-test
    pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se
    pval = 2 * (1 - norm.cdf(np.abs(z)))

    out["P_Value"] = pval
    out["Significant_95"] = out["P_Value"] < alpha

    # CI for (p2 - p1)
    zcrit = norm.ppf(1 - alpha / 2)
    se_diff = np.sqrt((p1 * (1 - p1)) / n1 + (p2 * (1 - p2)) / n2)
    out["CI_Diff_Low"] = (p2 - p1) - zcrit * se_diff
    out["CI_Diff_High"] = (p2 - p1) + zcrit * se_diff

    # flags (simple)
    out["Data_Flag"] = ""
    out.loc[(n1 < DEFAULT_THRESHOLDS.min_n_low) | (n2 < DEFAULT_THRESHOLDS.min_n_low), "Data_Flag"] = "Low sample"
    out.loc[
        ((n1 >= DEFAULT_THRESHOLDS.min_n_low) & (n1 < DEFAULT_THRESHOLDS.min_n_warn)) |
        ((n2 >= DEFAULT_THRESHOLDS.min_n_low) & (n2 < DEFAULT_THRESHOLDS.min_n_warn)),
        "Data_Flag"
    ] = out["Data_Flag"].replace("", "Limited sample")

    # display fields
    out["Control_Pct"] = out["Control_Prop"] * 100
    out["Exposed_Pct"] = out["Exposed_Prop"] * 100
    out["Lift_Pct"] = out["Lift_Rel"] * 100
    out["Diff_PctPts"] = out["Diff_Prop"] * 100
    out["CI_Low_PctPts"] = out["CI_Diff_Low"] * 100
    out["CI_High_PctPts"] = out["CI_Diff_High"] * 100

    # extra computed fields (optional, but useful)
    out["Total_Sample"] = out["Control Sample"] + out["Exposed Sample"]
    out["Z_Score"] = z
    out["Std_Error"] = se
    out["SE_Diff"] = se_diff
    out["Pooled_Prop"] = pooled

    return out
