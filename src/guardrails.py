"""
guardrails.py — Data Leakage & Bias Assertion Module
Run these checks at each stage of the pipeline to catch leakage early.
"""

import numpy as np
import pandas as pd
from typing import Any


# ─────────────────────────────────────────────
# LEAKAGE ASSERTIONS
# ─────────────────────────────────────────────

def assert_no_target_in_features(X: pd.DataFrame, target_col: str = "ChurnStatus") -> None:
    """Rule L1: ChurnStatus must never appear in feature matrix."""
    if target_col in X.columns:
        raise AssertionError(
            f"LEAKAGE: '{target_col}' found in feature matrix. "
            "Strip target before preprocessing."
        )


def assert_no_id_in_features(X: pd.DataFrame, id_col: str = "CustomerID") -> None:
    """Rule L6: CustomerID must not be a model feature."""
    if id_col in X.columns:
        raise AssertionError(
            f"LEAKAGE: '{id_col}' found in feature matrix. "
            "Drop ID columns before training."
        )


def assert_no_future_in_features(X: pd.DataFrame, time_col: str = "Month") -> None:
    """Rule L1: Month column must be removed or converted to ordinal before training."""
    if time_col in X.columns:
        raise AssertionError(
            f"LEAKAGE: Raw '{time_col}' column found in feature matrix. "
            "Drop or encode the time column before fitting."
        )


def assert_transformer_fit_on_train_only(
    transformer: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> None:
    """
    Rule L2: Verify transformer was fit on train only by checking that
    fitting on train+test would produce a different result.
    Raises AssertionError if transformer appears to have been fit on combined data.
    This is a heuristic check — works for scalers that store mean/scale.
    """
    # Check for RobustScaler / StandardScaler type attributes
    if hasattr(transformer, "center_"):
        combined = pd.concat([X_train, X_test], ignore_index=True)
        train_medians = X_train.median()
        combined_medians = combined.median()
        stored = pd.Series(transformer.center_, index=X_train.columns if hasattr(X_train, 'columns') else range(len(transformer.center_)))
        # If stored values match combined dataset, it was likely fit on combined
        if np.allclose(stored.values, combined_medians.values, rtol=0.01):
            if not np.allclose(stored.values, train_medians.values, rtol=0.01):
                raise AssertionError(
                    "LEAKAGE: Scaler appears to have been fit on combined train+test data. "
                    "Fit only on training data."
                )


def assert_imputation_from_train(
    X_train_before: pd.DataFrame,
    X_test_before: pd.DataFrame,
    X_test_after: pd.DataFrame,
    train_stats: dict,
) -> None:
    """
    Rule L3: Verify that imputed values in test match train statistics (not test statistics).
    Checks a sample of null positions.
    """
    for col in X_test_before.columns:
        if X_test_before[col].isna().any() and col in train_stats:
            null_positions = X_test_before[col].isna()
            imputed_values = X_test_after.loc[null_positions, col].unique()
            expected_fill = train_stats[col]
            if not all(np.isclose(v, expected_fill, rtol=0.01) for v in imputed_values if not pd.isna(v)):
                raise AssertionError(
                    f"LEAKAGE: Column '{col}' imputed with value {imputed_values} "
                    f"but expected train value {expected_fill:.4f}. "
                    "Use train statistics for imputation."
                )


def assert_psi_uses_train_bins(psi_func_result: float, col_name: str = "") -> None:
    """
    Rule L4: PSI must be computed using breakpoints from training distribution.
    This is a documentation assertion — call it after computing PSI to confirm the call was correct.
    Always pass psi_func_result from calculate_psi(train_col, test_col) not (test, train).
    PSI > 10 is physically impossible for a well-formed calculation — flag it.
    """
    if psi_func_result < 0:
        raise AssertionError(f"LEAKAGE/BUG: PSI for '{col_name}' is negative ({psi_func_result:.4f}). "
                             "PSI must be non-negative. Check bin computation uses train breakpoints.")
    if psi_func_result > 10:
        raise AssertionError(f"BUG: PSI for '{col_name}' is {psi_func_result:.2f} — unreasonably large. "
                             "Likely computing PSI(test, train) instead of PSI(train, test).")


def assert_no_duplicate_customers_in_eval(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id_col: str = "CustomerID",
) -> None:
    """
    Rule L7: Flag CustomerIDs that appear in both train and test.
    Does not remove them (we need predictions for all test rows) but warns.
    """
    overlap = set(train_df[id_col]) & set(test_df[id_col])
    if overlap:
        print(f"[GUARDRAIL WARNING] {len(overlap)} CustomerID(s) appear in both train and test: {overlap}")
        print("  Their test predictions may be influenced by memorization. Flagged for awareness.")


# ─────────────────────────────────────────────
# FORMAT / CASE GUARDRAILS
# ─────────────────────────────────────────────

KNOWN_CASE_DRIFT_COLS = ["Contract", "TransactionMode", "InternetType"]


def assert_case_normalized(df: pd.DataFrame, str_cols: list) -> None:
    """
    Rule F1: All string columns must be lowercase+stripped before encoding.
    Catches the confirmed drift where Contract = 'Month-to-Month' (train) vs 'month-to-month' (test).
    """
    for col in str_cols:
        if col not in df.columns:
            continue
        sample = df[col].dropna().head(100)
        mixed_case = sample.apply(lambda x: x != x.lower() if isinstance(x, str) else False).any()
        if mixed_case:
            raise AssertionError(
                f"FORMAT DRIFT RISK: Column '{col}' contains non-lowercase values. "
                "Apply .str.strip().str.lower() to both train and test before encoding."
            )


def assert_null_as_category_applied(df: pd.DataFrame, high_null_cols: list) -> None:
    """
    Rule F2: High-null columns must have NaNs filled with 'unknown' before encoding.
    PrioritySupport and DigitalInvoicing have very different null rates between train/test.
    """
    for col in high_null_cols:
        if col not in df.columns:
            continue
        if df[col].isna().any():
            raise AssertionError(
                f"LEAKAGE RISK: Column '{col}' still has {df[col].isna().sum()} NaN values. "
                "Fill with 'unknown' to treat missingness as a valid category."
            )


# ─────────────────────────────────────────────
# BIAS ASSERTIONS
# ─────────────────────────────────────────────

def assert_no_demographic_dominance(
    feature_importances: pd.Series,
    demographic_cols: list = None,
    threshold: float = 0.10,
) -> None:
    """
    Rule B1/B2: No single demographic or geographic feature should dominate feature importance.
    Raises warning (not error) if any demographic feature exceeds threshold% of total gain.
    """
    if demographic_cols is None:
        demographic_cols = [
            "UserGender", "Married", "Dependents", "YoungAdultFlag",
            "RetireeStatus", "Country", "State", "LocationCity",
            "UserAge", "AreaCode", "Latitude", "Longitude",
        ]
    total_importance = feature_importances.sum()
    if total_importance == 0:
        return
    for col in demographic_cols:
        if col in feature_importances.index:
            pct = feature_importances[col] / total_importance
            if pct > threshold:
                print(
                    f"[BIAS WARNING] '{col}' has {pct:.1%} of total feature importance "
                    f"(threshold={threshold:.0%}). Review for geographic/demographic bias."
                )


def assert_auprc_parity(
    y_true: pd.Series,
    y_pred: np.ndarray,
    group_col: pd.Series,
    min_group_size: int = 50,
    max_gap: float = 0.10,
) -> None:
    """
    Rule B1: AU-PRC should not vary dramatically across demographic groups.
    Raises warning if gap between best and worst group exceeds max_gap.
    """
    from sklearn.metrics import average_precision_score
    groups = group_col.unique()
    scores = {}
    for g in groups:
        mask = group_col == g
        if mask.sum() < min_group_size:
            continue
        try:
            scores[g] = average_precision_score(y_true[mask], y_pred[mask])
        except Exception:
            pass
    if not scores:
        return
    best, worst = max(scores.values()), min(scores.values())
    if (best - worst) > max_gap:
        print(f"[BIAS WARNING] AU-PRC gap across groups: {worst:.3f} to {best:.3f} (gap={best-worst:.3f} > {max_gap})")
        for g, s in sorted(scores.items(), key=lambda x: x[1]):
            print(f"  {group_col.name}={g}: AU-PRC={s:.3f}")


# ─────────────────────────────────────────────
# PIPELINE STAGE RUNNER
# ─────────────────────────────────────────────

def run_pre_processing_checks(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Run all guardrails BEFORE any preprocessing. Call at pipeline start."""
    print("[GUARDRAILS] Running pre-processing checks...")
    assert_no_duplicate_customers_in_eval(train_df, test_df)
    print("  [OK] CustomerID overlap check")
    print("[GUARDRAILS] Pre-processing checks complete.\n")


def run_post_feature_engineering_checks(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> None:
    """Run after feature engineering, before drift detection."""
    print("[GUARDRAILS] Running post-feature-engineering checks...")
    for df, name in [(X_train, "X_train"), (X_test, "X_test")]:
        assert_no_target_in_features(df)
        assert_no_id_in_features(df)
        assert_no_future_in_features(df)
        print(f"  [OK] {name}: no target/ID/time leakage")
    print("[GUARDRAILS] Feature engineering checks complete.\n")


def run_post_mitigation_checks(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    str_cols: list,
) -> None:
    """Run after drift mitigation, before model training."""
    print("[GUARDRAILS] Running post-mitigation checks...")
    assert_case_normalized(X_train, str_cols)
    assert_case_normalized(X_test, str_cols)
    print("  [OK] Case normalization verified")
    print("[GUARDRAILS] Post-mitigation checks complete.\n")


def run_post_training_checks(
    model,
    y_true_train: np.ndarray,
    y_pred_train: np.ndarray,
    y_true_test: np.ndarray,
    y_pred_test: np.ndarray,
    X_train: pd.DataFrame,
    test_df_with_demographics: pd.DataFrame,
) -> None:
    """Run after model training. Checks bias and sanity."""
    from sklearn.metrics import average_precision_score
    print("[GUARDRAILS] Running post-training checks...")

    # Sanity: AU-PRC > 0.3 (better than random given 28.8% positive rate)
    train_auprc = average_precision_score(y_true_train, y_pred_train)
    test_auprc = average_precision_score(y_true_test, y_pred_test)
    if train_auprc < 0.3:
        print(f"  [WARN] Train AU-PRC={train_auprc:.3f} is suspiciously low (baseline ~0.288)")
    if test_auprc < 0.3:
        print(f"  [WARN] Test AU-PRC={test_auprc:.3f} is suspiciously low")

    # Overfitting check
    if train_auprc - test_auprc > 0.15:
        print(f"  [WARN] Large train-test AU-PRC gap: {train_auprc:.3f} vs {test_auprc:.3f}. "
              "Possible overfitting — check feature engineering for leakage.")
    else:
        print(f"  [OK] AU-PRC gap acceptable: train={train_auprc:.3f}, test={test_auprc:.3f}")

    # Feature importance bias check
    if hasattr(model, 'feature_importances_'):
        importance = pd.Series(model.feature_importances_,
                               index=X_train.columns if hasattr(X_train, 'columns') else range(len(model.feature_importances_)))
        assert_no_demographic_dominance(importance)
        print("  [OK] Demographic dominance check complete")

    # Bias parity check by gender if available
    if 'UserGender' in test_df_with_demographics.columns:
        y_true_s = pd.Series(y_true_test)
        group_s = test_df_with_demographics['UserGender'].reset_index(drop=True)
        assert_auprc_parity(y_true_s, y_pred_test, group_s)
        print("  [OK] Gender parity check complete")

    print("[GUARDRAILS] Post-training checks complete.\n")
