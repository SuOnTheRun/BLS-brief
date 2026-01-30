import math
import json
import pandas as pd


# -----------------------------
# Math helpers
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
# Notes copy dictionary (single source)
# -----------------------------
NOTES_COPY = {
  "LOW_RELIABILITY": {
    "short": "Small sample — treat as a hint.",
    "full": "Only a small number of people answered this question. With small samples, results can swing easily, so treat this as a hint rather than a conclusion."
  },
  "DIRECTIONAL_SIGNAL": {
    "short": "Early signal — not fully confirmed.",
    "full": "This result points in a clear direction, but we don’t yet have enough evidence to be fully confident. This often happens when the number of responses is moderate rather than large."
  },
  "UNCLEAR_SIGNAL": {
    "short": "Could be random variation.",
    "full": "We can’t reliably tell whether this difference is real or just random variation. This does not mean nothing happened — it means we don’t have enough evidence yet."
  },
  "LIFT_HIDDEN_SMALL_BASE": {
    "short": "Lift hidden — baseline too small.",
    "full": "Relative change is hidden because the starting number is very small. When the baseline is tiny, lift can look much bigger than the real-world change and can be misleading. The difference out of 100 people is the clearest way to read this result."
  },
  "FLAT_RESULT": {
    "short": "Little to no change.",
    "full": "The exposed and control groups answered in very similar ways. This suggests the ads did not meaningfully change this metric during the measured period."
  },
  "NEGATIVE_RESULT": {
    "short": "Moved down vs control (learning signal).",
    "full": "Fewer people who saw the ads answered positively compared to those who did not. This can happen for many reasons and is a useful learning signal rather than a failure."
  },
  "AGGREGATED_RESULT": {
    "short": "Average across multiple results.",
    "full": "This value is an average across multiple periods or results. Individual waves may perform better or worse than this overall number."
  },
  "MIXED_PERFORMANCE": {
    "short": "Mixed results underneath.",
    "full": "This average combines both stronger and weaker results. The overall number shows direction, but underlying performance is mixed."
  },
  "DATA_MISSING": {
    "short": "Not enough responses to interpret.",
    "full": "Some results are not shown because there were not enough responses to interpret them reliably."
  },
  "FILTER_IMPACT": {
    "short": "Filters significantly narrow the view.",
    "full": "The current filters significantly narrow the data shown. Results may differ when viewing a broader time period or additional markets."
  }
}

NOTE_PRIORITY = [
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


# -----------------------------
# Core stats computation
# -----------------------------
def compute_all_metrics(
    df: pd.DataFrame,
    *,
    # Movable reliability thresholds
    GREAT_THRESHOLD: int = 300,
    GOOD_THRESHOLD: int = 100,
    DIRECTIONAL_THRESHOLD: int = 50,

    # Clarity thresholds
    CLEAR_P_THRESHOLD: float = 0.05,
    DIRECTIONAL_P_THRESHOLD: float = 0.10,

    # Lift visibility rules
    MIN_BASELINE_PERCENT: float = 5.0,
    MIN_LIFT_SAMPLE: int = 50,

    # Flat threshold (pp)
    FLAT_THRESHOLD_PP: float = 1.0,
) -> pd.DataFrame:
    """
    Computes full BLS stats + interpretation bands + notes.

    Required input columns:
      - Control Sample
      - Exposed Sample
      - Control Score
      - Exposed Score

    Scores can be 0–100 (percent) or 0–1 (proportion).

    Outputs include:
      - Control_Pct, Exposed_Pct, Diff_PctPts, Lift_Pct (may be NaN if hidden)
      - P_Value, CI_Low_PctPts, CI_High_PctPts
      - Reliability_N, Reliability_Band (Great/Good/Directional/Low)
      - Clarity_Band (Clear/Directional/Unclear)
      - Lift_Visible (bool)
      - Notes_Keys (json string list), Notes_Short, Notes_Full
      - Human_* helper fields for "out of 100" reading
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

    # -----------------------------
    # Core numeric outputs
    # -----------------------------
    d["Control_Pct"] = (p1 * 100.0)
    d["Exposed_Pct"] = (p2 * 100.0)

    d["Diff_PctPts"] = (diff * 100.0)
    d["Lift_Pct_raw"] = (lift * 100.0)

    d["Pooled_Prop"] = pooled
    d["Std_Error"] = se
    d["Z_Score"] = z
    d["P_Value"] = pval

    d["CI_Low_PctPts"] = (ci_low * 100.0)
    d["CI_High_PctPts"] = (ci_high * 100.0)

    # -----------------------------
    # Reliability bands (movable)
    # -----------------------------
    reliability_n = pd.concat([n1, n2], axis=1).min(axis=1)
    d["Reliability_N"] = reliability_n

    def _rel_band(mn):
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

    d["Reliability_Band"] = d["Reliability_N"].apply(_rel_band)

    # -----------------------------
    # Clarity bands (with reliability gate for Clear)
    # -----------------------------
    def _clar_band(pv, mn):
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

    d["Clarity_Band"] = [
        _clar_band(pv, mn) for pv, mn in zip(d["P_Value"].tolist(), d["Reliability_N"].tolist())
    ]

    # -----------------------------
    # Lift visibility rules
    # -----------------------------
    def _lift_visible(control_pct, control_n):
        try:
            control_pct = float(control_pct)
            control_n = float(control_n)
        except Exception:
            return False
        if control_pct != control_pct or control_n != control_n:
            return False
        if control_pct < MIN_BASELINE_PERCENT:
            return False
        if control_n < MIN_LIFT_SAMPLE:
            return False
        return True

    d["Lift_Visible"] = [
        bool(_lift_visible(cp, cn))
        for cp, cn in zip(d["Control_Pct"].tolist(), n1.tolist())
    ]

    d["Lift_Pct"] = d["Lift_Pct_raw"]
    d.loc[~d["Lift_Visible"], "Lift_Pct"] = float("nan")

    # -----------------------------
    # Human helper fields ("out of 100" + "out of n")
    # -----------------------------
    d["Human_Control_OutOf100"] = d["Control_Pct"].astype(float).round(1)
    d["Human_Exposed_OutOf100"] = d["Exposed_Pct"].astype(float).round(1)
    d["Human_Gap_OutOf100"] = d["Diff_PctPts"].astype(float).round(2)

    # "About X out of N" helper
    def _out_of_n(pct, n):
        try:
            pct = float(pct); n = float(n)
        except Exception:
            return ""
        if pct != pct or n != n or n <= 0:
            return ""
        x = int(round((pct / 100.0) * n))
        return f"~{x} out of {int(round(n))}"

    d["Human_Control_OutOfN"] = [
        _out_of_n(cp, cn) for cp, cn in zip(d["Control_Pct"].tolist(), n1.tolist())
    ]
    d["Human_Exposed_OutOfN"] = [
        _out_of_n(ep, en) for ep, en in zip(d["Exposed_Pct"].tolist(), n2.tolist())
    ]

    # -----------------------------
    # Notes engine (row-level)
    # -----------------------------
    def _notes_for_row(row) -> list:
        notes = []

        # Missing / invalid data guard
        if row.get("Control_Pct") != row.get("Control_Pct") or row.get("Exposed_Pct") != row.get("Exposed_Pct"):
            notes.append("DATA_MISSING")
            return notes

        # Reliability
        if row.get("Reliability_Band") == "Low":
            notes.append("LOW_RELIABILITY")

        # Clarity
        if row.get("Clarity_Band") == "Directional":
            notes.append("DIRECTIONAL_SIGNAL")
        elif row.get("Clarity_Band") == "Unclear":
            notes.append("UNCLEAR_SIGNAL")

        # Lift hidden
        if not bool(row.get("Lift_Visible", True)):
            notes.append("LIFT_HIDDEN_SMALL_BASE")

        # Flat / negative
        gap = float(row.get("Diff_PctPts", 0.0))
        if abs(gap) < float(FLAT_THRESHOLD_PP):
            notes.append("FLAT_RESULT")
        if gap < 0:
            notes.append("NEGATIVE_RESULT")

        # De-dup & priority order
        notes = list(dict.fromkeys(notes))
        notes_sorted = [k for k in NOTE_PRIORITY if k in notes]
        return notes_sorted

    keys = []
    shorts = []
    fulls = []
    for _, r in d.iterrows():
        nk = _notes_for_row(r)
        keys.append(nk)
        shorts.append(" • ".join([NOTES_COPY[k]["short"] for k in nk]) if nk else "")
        fulls.append("\n".join([NOTES_COPY[k]["full"] for k in nk]) if nk else "")

    d["Notes_Keys"] = [json.dumps(x) for x in keys]
    d["Notes_Short"] = shorts
    d["Notes_Full"] = fulls

    # Backward-compat (so older UI pieces don't break)
    # Map to old labels if you still reference `Reliability`
    map_old = {"Great": "High", "Good": "Medium", "Directional": "Directional", "Low": "Low"}
    d["Reliability"] = d["Reliability_Band"].map(map_old)

    return d
