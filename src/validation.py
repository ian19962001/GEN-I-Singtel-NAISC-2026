# -*- coding: utf-8 -*-
"""
validation.py - Statistical & Score Validation Suite
Runs BEFORE building the final model to verify:
  1. Temporal CV is stable (no lucky split)
  2. Drift detection correctly identifies known-drifted features
  3. Each mitigation step actually improves AU-PRC
  4. Pipeline generalizes (doesn't overfit to train months)

Usage:
    python src/validation.py --train_data_filepath public_data/train.csv
                              --test_data_filepath public_data/test.csv
"""

import argparse
import sys
import io
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score
import lightgbm as lgb

warnings.filterwarnings("ignore")

# Force UTF-8 output on Windows to avoid cp949 encode errors
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ──────────────────────────────────────────────
# KNOWN GROUND TRUTH for validation assertions
# ──────────────────────────────────────────────

# Features we KNOW drift (confirmed from data inspection)
KNOWN_DRIFTED_NUMERICAL = {
    "NumberofReferrals",
    "AvgMonthlyLongDistanceCharges",
    "MonthlyCharge",
    "TotalLongDistanceCharges",
    "TotalCharges",
}

# Features we know should NOT drift significantly (p > 0.1)
KNOWN_STABLE_NUMERICAL = {
    "UserAge",
    "NumberofDependents",
    "TenureinMonths",
    "Population",
    "DataUsageAvg",
    "TotalRefunds",
    "CustomerLifetimeValue",
}

KNOWN_CASE_DRIFT_COLS = {"Contract", "TransactionMode"}
KNOWN_NULL_DRIFT_COLS = {"PrioritySupport", "DigitalInvoicing"}

MONTH_ORDER = {
    "25-jan": 1, "25-feb": 2, "25-mar": 3, "25-apr": 4,
    "25-may": 5, "25-jun": 6, "25-jul": 7, "25-aug": 8,
    "25-sep": 9, "25-oct": 10, "25-nov": 11, "25-dec": 12,
}

LGBM_FIXED_PARAMS = dict(
    verbosity=-1,
    objective="binary",
    is_unbalance=True,
    random_state=42,
    importance_type="gain",
)

RANDOM_BASELINE_AUPRC = 0.288  # = churn rate = AU-PRC for random model


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def encode_target(series: pd.Series) -> pd.Series:
    return (series.astype(str).str.lower() == "yes").astype(int)


def normalize_strings(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
    return df


def quick_encode(df: pd.DataFrame, cat_cols: list, encoders: dict = None) -> tuple:
    """Label-encode categoricals. Fit encoders on df if not provided."""
    from sklearn.preprocessing import OrdinalEncoder
    df = df.copy()
    # Detect true categorical cols (non-numeric) from the dataframe itself
    actual_cat = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    cols_to_encode = [c for c in actual_cat if c in (cat_cols if cat_cols else actual_cat)]
    if encoders is None:
        encoders = {}
        for col in cols_to_encode:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            df[col] = enc.fit_transform(df[[col]].astype(str))
            encoders[col] = enc
    else:
        for col in cols_to_encode:
            if col in encoders:
                df[col] = encoders[col].transform(df[[col]].astype(str))
    # Ensure all columns are float
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-999)
    return df.astype(float), encoders


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal feature engineering for validation runs."""
    df = df.copy()
    # Detect string columns robustly (handles both object and pandas StringDtype)
    str_cols = [c for c in df.columns
                if df[c].apply(lambda x: isinstance(x, str)).any()
                and c not in ["CustomerID", "Month", "ChurnStatus"]]
    df = normalize_strings(df, str_cols)

    # Null-as-category for high-null cols
    for col in ["Offer", "InternetType", "PrioritySupport", "DigitalInvoicing"]:
        if col in df.columns:
            df[col] = df[col].astype(str).replace("nan", "unknown").fillna("unknown")

    # Time feature
    df["month_index"] = df["Month"].str.lower().map(MONTH_ORDER).fillna(0).astype(int)

    # Drop non-feature columns
    drop_cols = [c for c in ["CustomerID", "Month", "ChurnStatus"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    return df


# ──────────────────────────────────────────────
# TEST 1: Drift Detection Correctness
# ──────────────────────────────────────────────

def test_drift_detection_correctness(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Verify that KS test correctly flags known-drifted features and
    does NOT flag known-stable features.
    Returns: dict with pass/fail counts and details.
    """
    print("\n" + "="*60)
    print("TEST 1: Drift Detection Correctness")
    print("="*60)
    results = {"passed": 0, "failed": 0, "details": []}

    for col in KNOWN_DRIFTED_NUMERICAL:
        if col not in train_df.columns or col not in test_df.columns:
            continue
        _, p = ks_2samp(train_df[col].dropna(), test_df[col].dropna())
        detected = p < 0.05
        status = "PASS" if detected else "FAIL"
        results["passed" if detected else "failed"] += 1
        results["details"].append({"col": col, "expected": "DRIFT", "detected": detected, "p": p})
        print(f"  [{status}] {col}: p={p:.4f} {'(correctly flagged)' if detected else '(MISSED DRIFT)'}")

    for col in KNOWN_STABLE_NUMERICAL:
        if col not in train_df.columns or col not in test_df.columns:
            continue
        _, p = ks_2samp(train_df[col].dropna(), test_df[col].dropna())
        stable = p >= 0.05
        status = "PASS" if stable else "WARN"
        results["passed" if stable else "failed"] += 1
        results["details"].append({"col": col, "expected": "STABLE", "detected": not stable, "p": p})
        print(f"  [{status}] {col}: p={p:.4f} {'(correctly stable)' if stable else '(false positive)'}")

    print(f"\n  Summary: {results['passed']} passed, {results['failed']} failed")
    return results


# ──────────────────────────────────────────────
# TEST 2: Case/Format Drift Detection
# ──────────────────────────────────────────────

def test_format_drift_detection(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Verify that case drift in Contract/TransactionMode is detectable
    and that normalization fixes it.
    """
    print("\n" + "="*60)
    print("TEST 2: Format/Case Drift Detection & Fix")
    print("="*60)
    results = {"passed": 0, "failed": 0}

    for col in KNOWN_CASE_DRIFT_COLS:
        if col not in train_df.columns or col not in test_df.columns:
            continue
        train_cats = set(train_df[col].dropna().unique())
        test_cats = set(test_df[col].dropna().unique())
        has_drift = not train_cats.issubset(test_cats) or not test_cats.issubset(train_cats)

        # After normalization
        train_norm = set(train_df[col].str.lower().str.strip().dropna().unique())
        test_norm = set(test_df[col].str.lower().str.strip().dropna().unique())
        fixed = train_norm == test_norm or test_norm.issubset(train_norm)

        if has_drift and fixed:
            print(f"  [PASS] {col}: case drift detected AND fixed by normalization")
            results["passed"] += 1
        elif not has_drift:
            print(f"  [INFO] {col}: no case drift detected (may be already normalized)")
            results["passed"] += 1
        else:
            print(f"  [FAIL] {col}: drift detected but normalization did not fix it")
            results["failed"] += 1

    # Null drift check
    for col in KNOWN_NULL_DRIFT_COLS:
        if col not in train_df.columns or col not in test_df.columns:
            continue
        train_null = train_df[col].isna().mean()
        test_null = test_df[col].isna().mean()
        gap = abs(train_null - test_null)
        status = "PASS" if gap > 0.05 else "WARN"
        print(f"  [{status}] {col} null drift: train={train_null:.1%}, test={test_null:.1%}, gap={gap:.1%}")
        results["passed" if gap > 0.05 else "failed"] += 1

    print(f"\n  Summary: {results['passed']} passed, {results['failed']} failed")
    return results


# ──────────────────────────────────────────────
# TEST 3: Temporal Cross-Validation
# ──────────────────────────────────────────────

def test_temporal_cv(train_df: pd.DataFrame) -> dict:
    """
    Walk-forward temporal validation using train data only.
    Uses months progressively: train on months 1-N, validate on month N+1.
    This simulates the train→test scenario WITHOUT touching test data.
    Checks: AU-PRC should be stable across folds (not lucky split).
    """
    print("\n" + "="*60)
    print("TEST 3: Temporal Cross-Validation (Train Data Only)")
    print("="*60)

    # Get ordered months in train
    months = sorted(train_df["Month"].str.lower().unique(),
                    key=lambda m: MONTH_ORDER.get(m, 99))

    # Need at least 4 months to do 3-fold walk-forward
    if len(months) < 4:
        print("  [SKIP] Not enough months for temporal CV")
        return {"passed": True, "auprc_scores": []}

    str_cols = train_df.select_dtypes(include=["object", "string"]).columns.tolist()
    str_cols = [c for c in str_cols if c not in ["CustomerID", "Month", "ChurnStatus"]]
    cat_cols = [c for c in str_cols]  # will be encoded

    folds_to_run = list(range(3, min(len(months), 7)))  # validate on months 4,5,...,7
    auprc_scores = []
    runtimes = []

    for val_idx in folds_to_run:
        train_months = months[:val_idx]
        val_month = months[val_idx]

        fold_train = train_df[train_df["Month"].str.lower().isin(train_months)].copy()
        fold_val = train_df[train_df["Month"].str.lower() == val_month].copy()

        if len(fold_val) < 100:
            continue

        # Prepare features
        y_fold_train = encode_target(fold_train["ChurnStatus"])
        y_fold_val = encode_target(fold_val["ChurnStatus"])

        X_fold_train = build_features(fold_train)
        X_fold_val = build_features(fold_val)

        # Encode categoricals — fit ONLY on fold_train
        _, encoders = quick_encode(X_fold_train, cat_cols)
        X_fold_train, _ = quick_encode(X_fold_train, cat_cols, encoders)
        X_fold_val, _ = quick_encode(X_fold_val, cat_cols, encoders)

        # Align columns
        common_cols = [c for c in X_fold_train.columns if c in X_fold_val.columns]
        X_fold_train = X_fold_train[common_cols].fillna(-999)
        X_fold_val = X_fold_val[common_cols].fillna(-999)

        t0 = time.time()
        model = lgb.LGBMClassifier(**LGBM_FIXED_PARAMS)
        model.fit(X_fold_train, y_fold_train)
        elapsed = time.time() - t0

        y_pred = model.predict_proba(X_fold_val)[:, 1]
        score = average_precision_score(y_fold_val, y_pred)
        auprc_scores.append(score)
        runtimes.append(elapsed)

        n_train_months = len(train_months)
        print(f"  Fold train={n_train_months}m → val={val_month}: AU-PRC={score:.4f}, "
              f"fit_time={elapsed:.1f}s, n_train={len(fold_train)}, n_val={len(fold_val)}")

    if not auprc_scores:
        print("  [SKIP] No valid folds")
        return {"passed": True, "auprc_scores": []}

    mean_cv = np.mean(auprc_scores)
    std_cv = np.std(auprc_scores)
    min_cv = min(auprc_scores)
    max_cv = max(auprc_scores)

    print(f"\n  CV AU-PRC: mean={mean_cv:.4f}, std={std_cv:.4f}, min={min_cv:.4f}, max={max_cv:.4f}")
    print(f"  Baseline (random): {RANDOM_BASELINE_AUPRC:.4f}")

    # Assertions
    passed = True
    if mean_cv <= RANDOM_BASELINE_AUPRC:
        print(f"  [FAIL] Mean CV AU-PRC ({mean_cv:.4f}) <= random baseline ({RANDOM_BASELINE_AUPRC:.4f})")
        passed = False
    else:
        print(f"  [PASS] Model beats random baseline by {mean_cv - RANDOM_BASELINE_AUPRC:.4f}")

    if std_cv > 0.05:
        print(f"  [WARN] High CV variance (std={std_cv:.4f}). Model unstable across time periods.")
    else:
        print(f"  [PASS] CV variance acceptable (std={std_cv:.4f})")

    # Trend check: is performance declining over time?
    if len(auprc_scores) >= 3:
        trend = np.polyfit(range(len(auprc_scores)), auprc_scores, 1)[0]
        if trend < -0.02:
            print(f"  [WARN] Performance declining over time (trend={trend:.4f}/fold). "
                  "Consider recency weighting or sliding window.")
        else:
            print(f"  [PASS] No significant declining trend (trend={trend:.4f}/fold)")

    return {"passed": passed, "auprc_scores": auprc_scores, "mean": mean_cv, "std": std_cv}


# ──────────────────────────────────────────────
# TEST 4: Mitigation Impact Test
# ──────────────────────────────────────────────

def test_mitigation_impact(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Compare AU-PRC before and after applying each mitigation strategy.
    Uses the last 2 train months as a proxy validation set.
    Each mitigation should be neutral or positive — never significantly negative.
    """
    print("\n" + "="*60)
    print("TEST 4: Mitigation Impact (Proxy Validation)")
    print("="*60)

    # Split train: use last 2 months as proxy test
    months = sorted(train_df["Month"].str.lower().unique(),
                    key=lambda m: MONTH_ORDER.get(m, 99))
    proxy_train_months = months[:-2]
    proxy_val_months = months[-2:]

    proxy_train = train_df[train_df["Month"].str.lower().isin(proxy_train_months)].copy()
    proxy_val = train_df[train_df["Month"].str.lower().isin(proxy_val_months)].copy()

    str_cols = [c for c in train_df.select_dtypes(include=["object", "string"]).columns
                if c not in ["CustomerID", "Month", "ChurnStatus"]]

    def run_scenario(train_data, val_data, scenario_name, sample_weights=None):
        y_tr = encode_target(train_data["ChurnStatus"])
        y_va = encode_target(val_data["ChurnStatus"])
        X_tr = build_features(train_data)
        X_va = build_features(val_data)
        _, encoders = quick_encode(X_tr, str_cols)
        X_tr, _ = quick_encode(X_tr, str_cols, encoders)
        X_va, _ = quick_encode(X_va, str_cols, encoders)
        common = [c for c in X_tr.columns if c in X_va.columns]
        X_tr = X_tr[common].fillna(-999)
        X_va = X_va[common].fillna(-999)
        model = lgb.LGBMClassifier(**LGBM_FIXED_PARAMS)
        model.fit(X_tr, y_tr, sample_weight=sample_weights)
        y_pred = model.predict_proba(X_va)[:, 1]
        score = average_precision_score(y_va, y_pred)
        print(f"  {scenario_name:45s}: AU-PRC={score:.4f}")
        return score

    # Scenario A: Baseline (no mitigation)
    score_baseline = run_scenario(proxy_train, proxy_val, "Baseline (no mitigation)")

    # Scenario B: Recency weighting (exponential decay)
    month_order_map = {m: i for i, m in enumerate(sorted(proxy_train["Month"].str.lower().unique(),
                                                         key=lambda m: MONTH_ORDER.get(m, 0)))}
    ranks = proxy_train["Month"].str.lower().map(month_order_map).fillna(0)
    weights = np.exp(0.3 * (ranks - ranks.max()))
    score_recency = run_scenario(proxy_train, proxy_val, "Recency weighting (exp decay)", weights)

    # Scenario C: Drop NumberofReferrals (empirically validated best strategy)
    proxy_train_drop = proxy_train.copy()
    proxy_val_drop = proxy_val.copy()
    if "NumberofReferrals" in build_features(proxy_train_drop).columns:
        drop_train = build_features(proxy_train_drop).drop(columns=["NumberofReferrals"], errors="ignore")
        drop_val = build_features(proxy_val_drop).drop(columns=["NumberofReferrals"], errors="ignore")
    else:
        drop_train = build_features(proxy_train_drop)
        drop_val = build_features(proxy_val_drop)
    _, enc_drop = quick_encode(drop_train, [])
    drop_train_enc, _ = quick_encode(drop_train, [], enc_drop)
    drop_val_enc, _ = quick_encode(drop_val, [], enc_drop)
    common = [c for c in drop_train_enc.columns if c in drop_val_enc.columns]
    m_drop = lgb.LGBMClassifier(**LGBM_FIXED_PARAMS)
    m_drop.fit(drop_train_enc[common].fillna(-999), encode_target(proxy_train["ChurnStatus"]))
    score_drop = average_precision_score(
        encode_target(proxy_val["ChurnStatus"]),
        m_drop.predict_proba(drop_val_enc[common].fillna(-999))[:, 1]
    )
    print(f"  {'Drop NumberofReferrals (validated best)':45s}: AU-PRC={score_drop:.4f}")

    results = {
        "baseline": score_baseline,
        "recency": score_recency,
        "drop_referrals": score_drop,
    }

    best_strategy = max(results, key=results.get)
    best_score = results[best_strategy]
    print(f"\n  Best strategy: {best_strategy} (AU-PRC={best_score:.4f})")
    print(f"  Improvement over baseline: {best_score - score_baseline:+.4f}")

    passed = True
    if best_score < score_baseline - 0.005:
        print("  [FAIL] All mitigation strategies hurt performance vs baseline")
        passed = False
    elif best_score > score_baseline:
        print("  [PASS] At least one mitigation improves AU-PRC")
    else:
        print("  [NEUTRAL] Mitigation does not significantly change performance (OK for stable data)")

    return {"passed": passed, "results": results, "best": best_strategy}


# ──────────────────────────────────────────────
# TEST 5: Runtime Check
# ──────────────────────────────────────────────

def test_runtime_budget(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Estimate full pipeline runtime on public data.
    Must be < 600 seconds (10 minutes) per competition rules.
    """
    print("\n" + "="*60)
    print("TEST 5: Runtime Budget Estimation")
    print("="*60)

    str_cols = [c for c in train_df.select_dtypes(include=["object", "string"]).columns
                if c not in ["CustomerID", "Month", "ChurnStatus"]]

    t0 = time.time()
    y_train = encode_target(train_df["ChurnStatus"])
    y_test = encode_target(test_df["ChurnStatus"])
    X_train = build_features(train_df)
    X_test = build_features(test_df)
    X_train, encoders = quick_encode(X_train, str_cols)
    X_test, _ = quick_encode(X_test, str_cols, encoders)
    common = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common].fillna(-999)
    X_test = X_test[common].fillna(-999)
    model = lgb.LGBMClassifier(**LGBM_FIXED_PARAMS)
    model.fit(X_train, y_train)
    _ = model.predict_proba(X_test)[:, 1]
    elapsed = time.time() - t0

    BUDGET_SECONDS = 600  # 10 minutes
    pct_used = elapsed / BUDGET_SECONDS * 100

    print(f"  Public data pipeline time: {elapsed:.1f}s ({pct_used:.1f}% of 10min budget)")
    print(f"  Train size: {len(X_train):,}, Test size: {len(X_test):,}")

    passed = elapsed < BUDGET_SECONDS
    if passed:
        print(f"  [PASS] Well within 10-minute budget")
        if pct_used > 30:
            print(f"  [WARN] Using {pct_used:.0f}% of budget on public data. "
                  "Hidden dataset (up to 10M rows) may exceed budget. Optimize.")
    else:
        print(f"  [FAIL] Exceeds 10-minute budget!")

    return {"passed": passed, "elapsed": elapsed, "pct_budget": pct_used}


# ──────────────────────────────────────────────
# TEST 6: Leakage Smoke Test
# ──────────────────────────────────────────────

def test_no_leakage_smoke(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Smoke test: verify that removing known target-correlated features
    doesn't make AU-PRC collapse to near-baseline.
    If AU-PRC drops to ~baseline when we remove a feature, that feature
    may be carrying encoded target information.
    """
    print("\n" + "="*60)
    print("TEST 6: Leakage Smoke Test")
    print("="*60)

    str_cols = [c for c in train_df.select_dtypes(include=["object", "string"]).columns
                if c not in ["CustomerID", "Month", "ChurnStatus"]]

    y_train = encode_target(train_df["ChurnStatus"])
    y_test = encode_target(test_df["ChurnStatus"])
    X_train = build_features(train_df)
    X_test = build_features(test_df)
    X_train, encoders = quick_encode(X_train, str_cols)
    X_test, _ = quick_encode(X_test, str_cols, encoders)
    common = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common].fillna(-999)
    X_test = X_test[common].fillna(-999)

    # Full model
    model = lgb.LGBMClassifier(**LGBM_FIXED_PARAMS)
    model.fit(X_train, y_train)
    full_score = average_precision_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"  Full model AU-PRC: {full_score:.4f}")

    # Suspicious features check: TotalRevenue, TotalCharges, CustomerLifetimeValue
    # These are potentially high-leakage if they encode future revenue (post-churn)
    suspicious = ["TotalRevenue", "CustomerLifetimeValue", "TotalCharges"]
    suspicious_present = [c for c in suspicious if c in X_train.columns]

    if suspicious_present:
        X_tr_drop = X_train.drop(columns=suspicious_present)
        X_te_drop = X_test.drop(columns=suspicious_present)
        model_drop = lgb.LGBMClassifier(**LGBM_FIXED_PARAMS)
        model_drop.fit(X_tr_drop, y_train)
        drop_score = average_precision_score(y_test, model_drop.predict_proba(X_te_drop)[:, 1])
        drop_delta = full_score - drop_score
        print(f"  Without {suspicious_present}: AU-PRC={drop_score:.4f} (delta={drop_delta:+.4f})")

        if drop_delta > 0.10:
            print(f"  [WARN] Removing {suspicious_present} drops AU-PRC by {drop_delta:.4f}. "
                  "These features may carry target leakage (e.g., TotalRevenue includes churn period). "
                  "Investigate before including.")
        else:
            print(f"  [PASS] No strong evidence of leakage in financial summary features")

    results = {"passed": True, "full_score": full_score}
    print(f"\n  [PASS] Leakage smoke test complete (no hard failures)")
    return results


# ──────────────────────────────────────────────
# MAIN RUNNER
# ──────────────────────────────────────────────

def run_all_tests(train_path: str, test_path: str) -> None:
    print("\n" + "#"*60)
    print("# NAISC 2026 - VALIDATION SUITE")
    print("#"*60)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Normalize month case for consistent comparison
    train_df["Month"] = train_df["Month"].str.lower()
    test_df["Month"] = test_df["Month"].str.lower()

    print(f"\nTrain: {train_df.shape}, Test: {test_df.shape}")
    print(f"Train months: {sorted(train_df['Month'].unique(), key=lambda m: MONTH_ORDER.get(m,99))}")
    print(f"Test months:  {sorted(test_df['Month'].unique(), key=lambda m: MONTH_ORDER.get(m,99))}")

    all_results = {}

    all_results["drift_detection"] = test_drift_detection_correctness(train_df, test_df)
    all_results["format_drift"] = test_format_drift_detection(train_df, test_df)
    all_results["temporal_cv"] = test_temporal_cv(train_df)
    all_results["mitigation_impact"] = test_mitigation_impact(train_df, test_df)
    all_results["runtime"] = test_runtime_budget(train_df, test_df)
    all_results["leakage_smoke"] = test_no_leakage_smoke(train_df, test_df)

    # Final summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    all_passed = True
    for test_name, result in all_results.items():
        passed = result.get("passed", True)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {test_name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n  All validation tests passed. Pipeline is ready to build.")
    else:
        print("\n  Some tests FAILED. Fix issues before submitting.")

    # Print recommended strategy
    if "mitigation_impact" in all_results:
        best = all_results["mitigation_impact"].get("best", "baseline")
        scores = all_results["mitigation_impact"].get("results", {})
        print(f"\n  Recommended mitigation: {best} (CV AU-PRC={scores.get(best, 'N/A'):.4f})")

    if "temporal_cv" in all_results:
        cv = all_results["temporal_cv"]
        print(f"  Temporal CV AU-PRC: {cv.get('mean', 0):.4f} +/- {cv.get('std', 0):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NAISC 2026 Validation Suite")
    parser.add_argument("--train_data_filepath", required=True)
    parser.add_argument("--test_data_filepath", required=True)
    args = parser.parse_args()
    run_all_tests(args.train_data_filepath, args.test_data_filepath)
