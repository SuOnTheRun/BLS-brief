import io
import pandas as pd


TEMPLATE_COLUMNS = [
    # Base descriptors (recommended)
    "Market",
    "Brand",
    "Month Year",
    "KPI",
    "Category",
    "Study ID",
    "KPI Order",

    # Inputs only (must exist)
    "Control Sample",
    "Exposed Sample",
    "Control Score",
    "Exposed Score",
]


def build_template_bytes() -> bytes:
    df = pd.DataFrame(columns=TEMPLATE_COLUMNS)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")
