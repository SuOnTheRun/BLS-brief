from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from .notes import NOTES_PRIORITY


@dataclass
class Thresholds:
    # Reliability (movable bands)
    GREAT_THRESHOLD: int = 300
    GOOD_THRESHOLD: int = 100
    DIRECTIONAL_THRESHOLD: int = 50

    # Clarity
    CLEAR_P_THRESHOLD: float = 0.05
    DIRECTIONAL_P_THRESHOLD: float = 0.10

    # Lift visibility
    MIN_BASELINE_PERCENT: float = 5.0
    MIN_LIFT_SAMPLE: int = 50

    # Flat definition
    FLAT_THRESHOLD_PP: float = 1.0


def reliability_band(reliability_n: float, t: Thresholds) -> str:
    if pd.isna(reliability_n):
        return "Low"
    n = float(reliability_n)
    if n >= t.GREAT_THRESHOLD:
        return "Great"
    if n >= t.GOOD_THRESHOLD:
        return "Good"
    if n >= t.DIRECTIONAL_THRESHOLD:
        return "Directional"
    return "Low"


def clarity_band(p_value: float, reliability_n: float, t: Thresholds) -> str:
    if pd.isna(p_value):
        return "Unclear"
    pv = float(p_value)
    rn = float(reliability_n) if not pd.isna(reliability_n) else 0.0

    if pv <= t.CLEAR_P_THRESHOLD and rn >= t.GOOD_THRESHOLD:
        return "Clear"
    if pv <= t.DIRECTIONAL_P_THRESHOLD:
        return "Directional"
    return "Unclear"


def lift_hidden(control_pct: float, control_n: float, t: Thresholds) -> bool:
    if pd.isna(control_pct) or pd.isna(control_n):
        return True
    return (float(control_pct) < t.MIN_BASELINE_PERCENT) or (float(control_n) < t.MIN_LIFT_SAMPLE)


def evaluate_row(row: pd.Series, t: Thresholds, flags: Dict) -> Tuple[pd.Series, List[str]]:
    """
    Attaches:
      - reliability_n, reliability_band
      - clarity_band
      - lift_visible (bool)
      - notes[] (keys)
    """
    notes: List[str] = []

    # Data availability
    if flags.get("metric_missing", False):
        notes.append("DATA_MISSING")

    # Reliability
    reliability_n = min(row.get("Control Sample", np.nan), row.get("Exposed Sample", np.nan))
    rb = reliability_band(reliability_n, t)
    if rb == "Low":
        notes.append("LOW_RELIABILITY")

    # Clarity
    cb = clarity_band(row.get("P_Value", np.nan), reliability_n, t)
    if cb == "Directional":
        notes.append("DIRECTIONAL_SIGNAL")
    if cb == "Unclear":
        notes.append("UNCLEAR_SIGNAL")

    # Lift visibility
    lh = lift_hidden(row.get("Control_Pct", np.nan), row.get("Control Sample", np.nan), t)
    if lh:
        notes.append("LIFT_HIDDEN_SMALL_BASE")

    # Direction & magnitude
    gap_pp = row.get("Diff_PctPts", np.nan)
    if not pd.isna(gap_pp):
        if abs(float(gap_pp)) < t.FLAT_THRESHOLD_PP:
            notes.append("FLAT_RESULT")
        if float(gap_pp) < 0:
            notes.append("NEGATIVE_RESULT")

    # Filter impact (global flag can be attached at render-level too)
    if flags.get("filters_reduce_rows_significantly", False):
        notes.append("FILTER_IMPACT")

    # Aggregation flags if used
    if flags.get("metric_is_aggregated", False):
        notes.append("AGGREGATED_RESULT")
    if flags.get("aggregation_contains_positive_and_negative", False):
        notes.append("MIXED_PERFORMANCE")

    # Priority ordering + de-dup
    ordered = []
    seen = set()
    for k in NOTES_PRIORITY:
        if k in notes and k not in seen:
            ordered.append(k)
            seen.add(k)
    for k in notes:
        if k not in seen:
            ordered.append(k)
            seen.add(k)

    out = row.copy()
    out["reliability_n"] = reliability_n
    out["reliability_band"] = rb
    out["clarity_band"] = cb
    out["lift_visible"] = (not lh)
    out["notes"] = ordered

    return out, ordered


def attach_rules(df: pd.DataFrame, t: Thresholds, flags: Dict) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        rr, _ = evaluate_row(r, t, flags)
        rows.append(rr)
    return pd.DataFrame(rows)
