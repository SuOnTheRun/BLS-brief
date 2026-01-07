import numpy as np
from .config import STATE_LABELS

def _state(row, include_non_sig=True):
    lift = row.get("Lift_Pct", np.nan)
    sig = bool(row.get("Significant_95", False))
    flag = str(row.get("Data_Flag", "")).strip()

    if not include_non_sig:
        return "clear_up" if lift >= 0 else "clear_down"

    if sig:
        return "clear_up" if lift >= 0 else "clear_down"

    # Non-significant rows: "possible" only if magnitude is not tiny and sample isn't clearly low
    if np.isnan(lift):
        return "no_clear"

    if abs(lift) >= 8 and flag != "Low sample":
        return "possible_up" if lift > 0 else "possible_down"

    return "no_clear"

def _note(row):
    sig = bool(row.get("Significant_95", False))
    flag = str(row.get("Data_Flag", "")).strip()

    if not sig and flag:
        return f"Not definitive; {flag.lower()}."
    if not sig:
        return "Not definitive; treat as directional."
    if flag:
        return f"Clear result; {flag.lower()}."
    return "Clear result."

def _meaning_line(row):
    lift = row.get("Lift_Pct", np.nan)
    diff = row.get("Diff_PctPts", np.nan)
    kpi = str(row.get("KPI", "KPI"))

    if np.isnan(lift) or np.isnan(diff):
        return f"{kpi}: data is incomplete, so the result is not readable."

    if lift >= 0:
        if diff >= 0.8:
            return f"{kpi} moved up in the exposed group by {diff:.2f} points."
        return f"{kpi} is slightly higher in the exposed group."
    else:
        if abs(diff) >= 0.8:
            return f"{kpi} dropped in the exposed group by {abs(diff):.2f} points."
        return f"{kpi} is slightly lower in the exposed group."

def _decision_line(row, state_key):
    lift = row.get("Lift_Pct", np.nan)
    sig = bool(row.get("Significant_95", False))

    if np.isnan(lift):
        return "Do not use this as evidence until the input is fixed."

    if state_key in ("clear_up", "clear_down"):
        if sig and abs(lift) >= 10:
            return "Safe to cite as evidence. Use it to justify the next decision."
        if sig:
            return "Usable as evidence, but the size is modest. Pair with context."
        return "Direction is clear, but confidence is weaker than ideal."

    if state_key in ("possible_up", "possible_down"):
        return "Direction is plausible. Keep it in, but avoid strong claims."

    return "No clear change. Keep it for completeness, not as a headline."

def build_insight_cards(df, include_non_sig=True):
    cards = []
    for _, row in df.iterrows():
        skey = _state(row, include_non_sig=include_non_sig)
        cards.append({
            "state_key": skey,
            "state_label": STATE_LABELS[skey],
            "note": _note(row),
            "meaning": _meaning_line(row),
            "decision": _decision_line(row, skey),
        })
    return cards
