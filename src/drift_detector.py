# -*- coding: utf-8 -*-
"""
drift_detector.py - Drift detection, quantification, and reporting.

Techniques used:
  - Numerical : Kolmogorov-Smirnov test + PSI (bins from TRAIN quantiles only)
  - Categorical: Chi-squared test + categorical PSI
  - Severity  : Composite score weighting p-value significance + PSI effect size

Output: structured drift report consumed by drift_mitigator.py and main.py.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
from tabulate import tabulate

# Columns to skip during drift detection (expected to differ by design)
SKIP_DRIFT_COLS = {'month_index'}

# PSI severity thresholds
PSI_MODERATE = 0.10
PSI_SEVERE   = 0.25


# ── Numerical drift ───────────────────────────────────────────────────────────

def detect_numerical_drift(train_col: pd.Series, test_col: pd.Series) -> dict:
    """KS test between train and test distributions."""
    train_clean = train_col.dropna()
    test_clean  = test_col.dropna()
    if len(train_clean) == 0 or len(test_clean) == 0:
        return {'ks_stat': 0.0, 'p_value': 1.0, 'drifted': False}
    stat, p_value = ks_2samp(train_clean, test_clean)
    return {
        'ks_stat' : round(float(stat),    4),
        'p_value' : round(float(p_value), 4),
        'drifted' : bool(p_value < 0.05),
    }


def calculate_psi_numerical(
    train_col: pd.Series,
    test_col:  pd.Series,
    bins:      int = 10,
) -> float:
    """
    Population Stability Index for numerical features.
    Breakpoints derived from TRAIN quantiles only (Rule L4 — no leakage).
    PSI < 0.10 : stable
    PSI 0.10-0.25 : moderate drift
    PSI > 0.25    : severe drift
    """
    train_arr = train_col.dropna().values
    test_arr  = test_col.dropna().values
    if len(train_arr) == 0 or len(test_arr) == 0:
        return 0.0

    breakpoints = np.unique(np.quantile(train_arr, np.linspace(0, 1, bins + 1)))
    if len(breakpoints) < 2:
        return 0.0

    expected = np.histogram(train_arr, bins=breakpoints)[0] / len(train_arr)
    actual   = np.histogram(test_arr,  bins=breakpoints)[0] / len(test_arr)

    # Avoid log(0)
    expected = np.where(expected == 0, 1e-6, expected)
    actual   = np.where(actual   == 0, 1e-6, actual)

    psi = float(np.sum((actual - expected) * np.log(actual / expected)))
    return round(max(psi, 0.0), 4)   # PSI is always non-negative


# ── Categorical drift ─────────────────────────────────────────────────────────

def detect_categorical_drift(train_col: pd.Series, test_col: pd.Series) -> dict:
    """Chi-squared test for categorical distribution shift."""
    all_cats    = set(train_col.dropna().unique()) | set(test_col.dropna().unique())
    train_counts = train_col.value_counts().reindex(all_cats, fill_value=0)
    test_counts  = test_col.value_counts().reindex(all_cats, fill_value=0)
    new_cats = set(test_col.dropna().unique()) - set(train_col.dropna().unique())

    if train_counts.sum() == 0 or test_counts.sum() == 0:
        return {'chi2_stat': 0.0, 'p_value': 1.0, 'drifted': False,
                'new_categories': new_cats}

    contingency = np.array([train_counts.values, test_counts.values])
    try:
        chi2, p_value, _, _ = chi2_contingency(contingency)
    except Exception:
        chi2, p_value = 0.0, 1.0

    return {
        'chi2_stat'      : round(float(chi2),    4),
        'p_value'        : round(float(p_value), 4),
        'drifted'        : bool(p_value < 0.05 or len(new_cats) > 0),
        'new_categories' : new_cats,
    }


def calculate_psi_categorical(train_col: pd.Series, test_col: pd.Series) -> float:
    """PSI for categorical features (category-level frequency comparison)."""
    all_cats = set(train_col.dropna().unique()) | set(test_col.dropna().unique())
    if not all_cats:
        return 0.0
    exp = train_col.value_counts(normalize=True).reindex(all_cats, fill_value=1e-6)
    act = test_col.value_counts(normalize=True).reindex(all_cats, fill_value=1e-6)
    psi = float(np.sum((act - exp) * np.log(act / exp)))
    return round(max(psi, 0.0), 4)


# ── Composite severity score ──────────────────────────────────────────────────

def drift_severity_score(p_value: float, psi: float) -> float:
    """
    Composite 0-1 drift severity.
    40% from statistical significance, 60% from PSI effect size.
    """
    p_score   = 1.0 - min(float(p_value), 1.0)
    psi_score = min(float(psi) / PSI_SEVERE, 1.0)
    return round(0.4 * p_score + 0.6 * psi_score, 4)


# ── Drift description ─────────────────────────────────────────────────────────

def classify_drift_type(
    train_col:    pd.Series,
    test_col:     pd.Series,
    is_numerical: bool,
    new_cats:     set = None,
) -> str:
    """Human-readable drift description for the output table."""
    if new_cats:
        sample = list(new_cats)[:2]
        return f'New categories in test set: {sample}'

    if is_numerical:
        try:
            train_mean, test_mean = float(train_col.mean()), float(test_col.mean())
            train_max,  test_max  = float(train_col.max()),  float(test_col.max())
            if test_max > train_max * 1.5:
                return (f'Range explosion: train max={train_max:.1f}, '
                        f'test max={test_max:.1f}')
            direction = 'higher' if test_mean > train_mean else 'lower'
            return (f'Distribution shifted {direction} '
                    f'(train mean={train_mean:.2f}, test mean={test_mean:.2f})')
        except Exception:
            return 'Numerical distribution changed in test set'

    # Categorical
    try:
        train_top = train_col.value_counts().index[0]
        test_top  = test_col.value_counts().index[0]
        if train_top != test_top:
            return (f'Top category changed: '
                    f'train="{train_top}", test="{test_top}"')
    except Exception:
        pass
    return 'Category frequency distribution changed in test set'


# ── Full drift report ─────────────────────────────────────────────────────────

def build_drift_report(X_train: pd.DataFrame, X_test: pd.DataFrame) -> list:
    """
    Run drift detection on all shared columns.
    Returns list of dicts for drifted features, sorted by severity descending.

    Each dict keys:
      column, col_type, drift_description, psi, severity, p_value,
      is_numerical, mitigation (empty string — filled by mitigator)
    """
    report = []

    for col in X_train.columns:
        if col in SKIP_DRIFT_COLS:
            continue
        if col not in X_test.columns:
            continue

        is_num = pd.api.types.is_numeric_dtype(X_train[col])

        if is_num:
            det      = detect_numerical_drift(X_train[col], X_test[col])
            psi      = calculate_psi_numerical(X_train[col], X_test[col])
            desc     = classify_drift_type(X_train[col], X_test[col], True)
            col_type = str(X_train[col].dtype)
            new_cats = set()
        else:
            det      = detect_categorical_drift(X_train[col], X_test[col])
            psi      = calculate_psi_categorical(X_train[col], X_test[col])
            desc     = classify_drift_type(X_train[col], X_test[col], False,
                                           det.get('new_categories', set()))
            col_type = 'object'
            new_cats = det.get('new_categories', set())

        severity = drift_severity_score(det['p_value'], psi)

        # Include in report if statistically drifted OR PSI exceeds moderate threshold
        if det['drifted'] or psi > PSI_MODERATE:
            report.append({
                'column'           : col,
                'col_type'         : col_type,
                'drift_description': desc,
                'psi'              : psi,
                'severity'         : severity,
                'p_value'          : det['p_value'],
                'is_numerical'     : is_num,
                'new_categories'   : new_cats,
                'mitigation'       : '',   # filled by mitigator
            })

    return sorted(report, key=lambda x: x['severity'], reverse=True)


# ── Console output ────────────────────────────────────────────────────────────

def print_drift_table(drift_report: list) -> None:
    """
    Print the required Drift Detection & Mitigation Summary table.
    Format: pipe-style table with 4 columns.
    """
    if not drift_report:
        print("No significant drift detected.")
        return

    rows = []
    for r in drift_report:
        # Truncate long descriptions to keep table readable
        desc = r['drift_description']
        if len(desc) > 65:
            desc = desc[:62] + '...'
        mitigation = r['mitigation'] if r['mitigation'] else 'None'
        rows.append([r['column'], r['col_type'], desc, mitigation])

    headers = [
        'Columns with Drift',
        'Column Type',
        'Drift Description',
        'Drift Mitigation',
    ]
    print(tabulate(rows, headers=headers, tablefmt='pipe'))


def psi_severity_label(psi: float) -> str:
    """Return human-readable PSI severity label."""
    if psi < PSI_MODERATE:
        return 'Stable'
    if psi < PSI_SEVERE:
        return 'Moderate'
    return 'Severe'
