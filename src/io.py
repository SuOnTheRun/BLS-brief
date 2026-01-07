import io
import pandas as pd
from .config import REQUIRED_BASE_COLS, SCORE_COLS, PROP_COLS

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
    df = _normalize_columns(df)
    cols = set(df.columns)

    missing_base = [c for c in REQUIRED_BASE_COLS if c not in cols]
    has_scores = all(c in cols for c in SCORE_COLS)
    has_props = all(c in cols for c in PROP_COLS)

    return {
        "ok": (len(missing_base) == 0) and (has_scores or has_props),
        "missing_base": missing_base,
        "has_scores": has_scores,
        "has_props": has_props,
        "columns": list(df.columns),
    }

def make_brand_alias_map(brands) -> dict:
    brands = [b for b in brands if str(b).strip() != "" and pd.notna(b)]
    unique = sorted(set(map(str, brands)))
    return {b: f"Brand {chr(65+i)}" for i, b in enumerate(unique)}

def apply_aliases(df: pd.DataFrame, alias_map: dict) -> pd.DataFrame:
    out = df.copy()
    out["Brand_Alias"] = out["Brand"].astype(str).map(alias_map).fillna("Brand ?")
    return out
