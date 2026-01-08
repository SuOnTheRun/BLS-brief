import io
import pandas as pd
from .config import REQUIRED_BASE_COLS, REQUIRED_SCORE_COLS, OPTIONAL_INPUT_COLS, ALLOWED_INPUT_COLS

def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    content = uploaded_file.getvalue()

    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(content))
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(io.BytesIO(content))
    raise ValueError("Please upload a CSV or XLSX file.")

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def validate_input(df: pd.DataFrame) -> dict:
    """
    Enforces:
      - REQUIRED_BASE_COLS
      - REQUIRED_SCORE_COLS (Control Score, Exposed Score)
    Allowed optional:
      - Study ID, KPI Order
    Everything else is "extra / computed".
    """
    df = _normalize_columns(df)
    cols = list(df.columns)
    colset = set(cols)

    missing_base = [c for c in REQUIRED_BASE_COLS if c not in colset]
    missing_scores = [c for c in REQUIRED_SCORE_COLS if c not in colset]

    extras = [c for c in cols if c not in set(ALLOWED_INPUT_COLS)]

    return {
        "ok": (len(missing_base) == 0 and len(missing_scores) == 0),
        "missing_base": missing_base,
        "missing_scores": missing_scores,
        "extras": extras,
        "columns": cols,
        "allowed_inputs": ALLOWED_INPUT_COLS
    }

def take_only_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps only the allowed input surface, drops everything else.
    """
    df = _normalize_columns(df)
    keep = [c for c in ALLOWED_INPUT_COLS if c in df.columns]
    return df.loc[:, keep].copy()
