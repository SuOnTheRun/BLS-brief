from dataclasses import dataclass

@dataclass(frozen=True)
class Thresholds:
    alpha: float = 0.05          # default 95%
    min_n_warn: int = 120        # warning threshold (not exclusion)
    min_n_low: int = 80          # stronger warning
    effect_small: float = 0.10   # reserved for future (Cohen's h)
    effect_medium: float = 0.30
    effect_large: float = 0.50

DEFAULT_THRESHOLDS = Thresholds()

# Required "input surface" columns.
REQUIRED_BASE_COLS = [
    "Month Year", "Brand", "Category", "Market", "KPI",
    "Control Sample", "Exposed Sample"
]

# The app accepts either:
# A) scores as percentages (Control Score, Exposed Score), OR
# B) proportions (Control_Prop, Exposed_Prop)
SCORE_COLS = ["Control Score", "Exposed Score"]
PROP_COLS = ["Control_Prop", "Exposed_Prop"]

STATE_LABELS = {
    "clear_up": "Clear increase",
    "clear_down": "Clear decline",
    "possible_up": "Possible increase",
    "possible_down": "Possible decline",
    "no_clear": "No clear change",
}
