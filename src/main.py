# -*- coding: utf-8 -*-
"""
main.py - Entry point for the NAISC 2026 Adaptive Drift Intelligence pipeline.

Usage:
    python ./src/main.py --train_data_filepath <path> --test_data_filepath <path>

Required console outputs:
    1. Drift Detection & Mitigation Summary table
    2. Time Taken (s) table
    3. Model Performance (AU-PRC) table

Required output files (written to project root):
    prediction.csv  - CustomerID, probability_score for all test rows
    model.joblib    - trained LightGBM model post-mitigation
"""

import argparse
import io
import os
import sys
import time

# ── Fix Windows console encoding (cp949 cannot encode some characters) ────────
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding='utf-8', errors='replace'
    )

# ── Ensure src/ modules are importable regardless of working directory ────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from feature_engineer import (
    encode_target,
    preprocess,
    fit_encoder,
    apply_encoder,
    align_to_training_columns,
)
from drift_detector import (
    build_drift_report,
    print_drift_table,
)
from drift_mitigator import (
    two_pass_train,
    print_runtime,
    print_performance,
)
from guardrails import (
    run_pre_processing_checks,
    run_post_feature_engineering_checks,
)


# ── Paths (always relative to project root, not src/) ────────────────────────
ROOT_DIR        = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PREDICTION_PATH = os.path.join(ROOT_DIR, 'prediction.csv')
MODEL_PATH      = os.path.join(ROOT_DIR, 'model.joblib')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='NAISC 2026 Adaptive Drift Intelligence Pipeline'
    )
    parser.add_argument('--train_data_filepath', required=True,
                        help='Path to training CSV')
    parser.add_argument('--test_data_filepath',  required=True,
                        help='Path to test CSV')
    args = parser.parse_args()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    try:
        train_df = pd.read_csv(args.train_data_filepath)
        test_df  = pd.read_csv(args.test_data_filepath)
    except Exception as e:
        print(f'ERROR loading data: {e}', file=sys.stderr)
        sys.exit(1)

    # ── 2. Pre-processing guardrails ──────────────────────────────────────────
    run_pre_processing_checks(train_df, test_df)

    # ── 3. Encode target (strip from features immediately) ────────────────────
    y_train = encode_target(train_df['ChurnStatus'])

    has_test_labels = 'ChurnStatus' in test_df.columns
    y_test = encode_target(test_df['ChurnStatus']) if has_test_labels else None

    # Preserve CustomerID for prediction.csv output
    test_ids = test_df['CustomerID'].copy()

    # ── 4. Preprocess (normalize, fill nulls, time feature, drop ID/Month/Target)
    X_train_raw = preprocess(train_df)
    X_test_raw  = preprocess(test_df)

    # ── 5. Build drift report on pre-encoded data (categoricals still strings)
    #    Timer starts here — detection is part of the measured pipeline
    t_start = time.time()

    drift_report = build_drift_report(X_train_raw, X_test_raw)
    print_drift_table(drift_report)
    print()

    # ── 6. Encode categoricals (fit on train ONLY) ────────────────────────────
    enc, cat_cols = fit_encoder(X_train_raw)
    X_train = apply_encoder(X_train_raw, enc, cat_cols).fillna(-999)
    X_test  = apply_encoder(X_test_raw,  enc, cat_cols).fillna(-999)

    # ── 7. Align test to training column set ──────────────────────────────────
    train_medians = X_train.median().to_dict()
    X_test = align_to_training_columns(
        X_test, list(X_train.columns), train_medians
    )

    # ── 8. Post-feature-engineering guardrails ────────────────────────────────
    run_post_feature_engineering_checks(X_train, X_test)

    # ── 9. Two-pass drift mitigation + training ───────────────────────────────
    final_model, X_train_clean, X_test_clean, dropped_cols = two_pass_train(
        X_train, y_train, X_test, drift_report
    )

    t_elapsed = time.time() - t_start

    # ── 10. Print runtime ─────────────────────────────────────────────────────
    print_runtime(t_elapsed)
    print()

    # ── 11. Evaluate and print AU-PRC ─────────────────────────────────────────
    train_probs = final_model.predict_proba(X_train_clean)[:, 1]
    test_probs  = final_model.predict_proba(X_test_clean)[:, 1]

    train_auprc = float(average_precision_score(y_train, train_probs))
    test_auprc  = (
        float(average_precision_score(y_test, test_probs))
        if y_test is not None else 0.0
    )

    print_performance(train_auprc, test_auprc)
    print()

    # ── 12. Save prediction.csv ───────────────────────────────────────────────
    pred_df = pd.DataFrame({
        'CustomerID'       : test_ids.values,
        'probability_score': test_probs,
    })
    pred_df.to_csv(PREDICTION_PATH, index=False)

    # ── 13. Save model.joblib ─────────────────────────────────────────────────
    joblib.dump(final_model, MODEL_PATH)

    print(f'Saved: {PREDICTION_PATH}')
    print(f'Saved: {MODEL_PATH}')

    if dropped_cols:
        print(f'Dropped features: {dropped_cols}')


if __name__ == '__main__':
    main()
