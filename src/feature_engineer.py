# -*- coding: utf-8 -*-
"""
feature_engineer.py - Preprocessing and feature engineering.

Rules enforced here:
  - All transformers fit on TRAIN only, applied to test (L2)
  - Target stripped before any feature ops (L1)
  - NaN imputed with train statistics only (L3)
  - Case normalization applied before encoding (F1)
  - High-null columns treated as category (F2)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# ── Month ordinal mapping ─────────────────────────────────────────────────────
MONTH_ORDER = {
    '25-jan': 1,  '25-feb': 2,  '25-mar': 3,  '25-apr': 4,
    '25-may': 5,  '25-jun': 6,  '25-jul': 7,  '25-aug': 8,
    '25-sep': 9,  '25-oct': 10, '25-nov': 11, '25-dec': 12,
}

# Columns where NaN is a meaningful signal (null rate shifts between train/test)
HIGH_NULL_COLS = ['Offer', 'InternetType', 'PrioritySupport', 'DigitalInvoicing']

# Columns to always drop before modelling
DROP_COLS = ['CustomerID', 'Month', 'ChurnStatus']


# ── Target encoding ───────────────────────────────────────────────────────────

def encode_target(series: pd.Series) -> pd.Series:
    """Handles 'Yes'/'No' strings, 1/0 integers, and pandas StringDtype."""
    return (series.astype(str).str.strip().str.lower() == 'yes').astype(int)


# ── String normalization ──────────────────────────────────────────────────────

def normalize_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lowercase and strip all string columns (except ID / time / target).
    Fixes confirmed case drift: Contract, TransactionMode.
    """
    df = df.copy()
    skip = set(DROP_COLS)
    for col in df.columns:
        if col in skip:
            continue
        # Detect string columns robustly (handles both object and StringDtype)
        if df[col].apply(lambda x: isinstance(x, str)).any():
            df[col] = df[col].astype(str).str.strip().str.lower()
    return df


# ── Null-as-category ──────────────────────────────────────────────────────────

def fill_nulls_as_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace NaN with 'unknown' for high-null columns.
    Preserves the missingness signal as a valid category.
    Confirmed null-rate drift: PrioritySupport 17.6%→29.3%, DigitalInvoicing 17.3%→0%.
    """
    df = df.copy()
    for col in HIGH_NULL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).replace('nan', 'unknown').fillna('unknown')
    return df


# ── Time feature ──────────────────────────────────────────────────────────────

def extract_time_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Month string to ordinal integer (month_index).
    Falls back to sorted-rank ordering for unknown month formats.
    """
    df = df.copy()
    if 'Month' not in df.columns:
        df['month_index'] = 0
        return df

    normalized = df['Month'].astype(str).str.strip().str.lower()
    result = normalized.map(MONTH_ORDER)

    # Fallback: rank unknown formats by sorted order
    if result.isna().any():
        unique_months = sorted(normalized.dropna().unique())
        fallback = {m: i + 1 for i, m in enumerate(unique_months)}
        result = result.fillna(normalized.map(fallback))

    df['month_index'] = result.fillna(0).astype(int)
    return df


# ── Main preprocessing ────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline (no encoding yet — keeps categoricals as strings
    so drift detection can run on original category values).

    Steps:
      1. Normalize string case
      2. Fill high-null cols as 'unknown'
      3. Extract month_index
      4. Drop CustomerID, Month, ChurnStatus
    """
    df = normalize_strings(df)
    df = fill_nulls_as_category(df)
    df = extract_time_feature(df)
    drop = [c for c in DROP_COLS if c in df.columns]
    return df.drop(columns=drop)


# ── Categorical encoder (fit on train ONLY) ───────────────────────────────────

def fit_encoder(X_train: pd.DataFrame) -> tuple:
    """
    Fit OrdinalEncoder on training data only.
    Returns (encoder, list_of_cat_cols).
    Unknown categories at inference get value -1 (LightGBM handles natively).
    """
    cat_cols = [
        c for c in X_train.columns
        if not pd.api.types.is_numeric_dtype(X_train[c])
    ]
    if not cat_cols:
        return None, []
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    enc.fit(X_train[cat_cols].astype(str))
    return enc, cat_cols


def apply_encoder(df: pd.DataFrame, enc, cat_cols: list) -> pd.DataFrame:
    """
    Apply fitted encoder to a dataframe.
    Coerces all columns to float64 (required by LightGBM).
    Handles columns that may be missing in the target dataframe.
    """
    df = df.copy()
    present = [c for c in cat_cols if c in df.columns]
    if enc is not None and present:
        df[present] = enc.transform(df[present].astype(str))
    # Coerce to float — fills any remaining non-numeric as NaN then -999
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.astype(float)


# ── Column alignment (hidden dataset robustness) ──────────────────────────────

def align_to_training_columns(
    X_test: pd.DataFrame,
    train_columns: list,
    train_medians: dict,
) -> pd.DataFrame:
    """
    Align test/hidden dataset to the exact columns seen during training.
      - Extra columns in test  → dropped
      - Missing columns in test → filled with train median (-999 fallback)
      - Column order            → matched to training exactly
    """
    X_test = X_test.copy()
    for col in train_columns:
        if col not in X_test.columns:
            X_test[col] = train_medians.get(col, -999)
    return X_test[train_columns]
