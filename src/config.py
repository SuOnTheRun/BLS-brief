from dataclasses import dataclass

@dataclass(frozen=True)
class Thresholds:
    alpha: float = 0.05          # 95%
    min_n_warn: int = 120
    min_n_low: int = 80

DEFAULT_THRESHOLDS = Thresholds()

# Base columns people must provide
REQUIRED_BASE_COLS = [
    "Month Year",
    "Brand",
    "Category",
    "Market",
    "KPI",
    "Control Sample",
    "Exposed Sample",
]

# We will enforce SCORES as the required input surface (not props)
REQUIRED_SCORE_COLS = ["Control Score", "Exposed Score"]

# Optional inputs (allowed, but not required)
OPTIONAL_INPUT_COLS = ["Study ID", "KPI Order"]

# Full allowed input surface = base + required scores + optional inputs
ALLOWED_INPUT_COLS = REQUIRED_BASE_COLS + REQUIRED_SCORE_COLS + OPTIONAL_INPUT_COLS

STATE_LABELS = {
    "clear_up": "Clear increase",
    "clear_down": "Clear decline",
    "possible_up": "Possible increase",
    "possible_down": "Possible decline",
    "no_clear": "No clear change",
}
