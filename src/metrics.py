import numpy as np
import pandas as pd
from scipy.stats import norm
from .config import DEFAULT_THRESHOLDS

def _to_float_pct(x):
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

    out["Control Sample"] = pd.to_numeric(out["Control Sample"], errors="coerce")
    out["Exposed Sample"] = pd.to_numeric(out["Exposed Sample"], errors="coerce")

    # Prefer explicit proportions if present, else derive from Score columns
    if "Control_Prop" in out.columns and "Exposed_Prop" in out.columns:
        out["Control_Prop"] = pd.to_numeric(out["Control_Prop"], errors="coerce")
        out["Exposed_Prop"] = pd.to_numeric(out["Exposed_Prop"], errors="coerce")
    else:
        out["Control_Prop"] = out.get("Control Score", np.nan).apply(_to_float_pct)
        out["Exposed_Prop"] = out.get("Exposed Score", np.nan).apply(_to_float_pct)

    out["Diff_Prop"] = out["Exposed_Prop"] - out["Control_Prop"]

    out["Lift_Rel"] = np.where(
        out["Control_Prop"].astype(float) == 0,
        np.nan,
        (out["Exposed_Prop"] - out["Control_Prop"]) / out["Control_Prop"]
    )

    # Two-proportion z-test
    n1 = out["Control Sample"].astype(float)
    n2 = out["Exposed Sample"].astype(float)
    p1 = out["Control_Prop"].astype(float)
    p2 = out["Exposed_Prop"].astype(float)

    pooled = (p1*n1 + p2*n2) / (n1 + n2)
    se = np.sqrt(pooled * (1 - pooled) * (1/n1 + 1/n2))

    z = (p2 - p1) / se
    pval = 2 * (1 - norm.cdf(np.abs(z)))

    out["P_Value"] = pval
    out["Significant_95"] = out["P_Value"] < alpha

    # 95% CI for difference in proportions (p2 - p1)
    zcrit = norm.ppf(1 - alpha/2)
    se_diff = np.sqrt((p1*(1-p1))/n1 + (p2*(1-p2))/n2)
    out["CI_Diff_Low"] = (p2 - p1) - zcrit * se_diff
    out["CI_Diff_High"] = (p2 - p1) + zcrit * se_diff

    # Simple reliability flags (warnings only)
    out["Data_Flag"] = ""
    out.loc[(n1 < DEFAULT_THRESHOLDS.min_n_low) | (n2 < DEFAULT_THRESHOLDS.min_n_low), "Data_Flag"] = "Low sample"
    out.loc[
        ((n1 >= DEFAULT_THRESHOLDS.min_n_low) & (n1 < DEFAULT_THRESHOLDS.min_n_warn)) |
        ((n2 >= DEFAULT_THRESHOLDS.min_n_low) & (n2 < DEFAULT_THRESHOLDS.min_n_warn)),
        "Data_Flag"
    ] = out["Data_Flag"].replace("", "Limited sample")

    # Display fields
    out["Control_Pct"] = out["Control_Prop"] * 100
    out["Exposed_Pct"] = out["Exposed_Prop"] * 100
    out["Lift_Pct"] = out["Lift_Rel"] * 100
    out["Diff_PctPts"] = out["Diff_Prop"] * 100
    out["CI_Low_PctPts"] = out["CI_Diff_Low"] * 100
    out["CI_High_PctPts"] = out["CI_Diff_High"] * 100

    return out
