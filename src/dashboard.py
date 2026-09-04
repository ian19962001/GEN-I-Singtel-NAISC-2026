# -*- coding: utf-8 -*-
"""
dashboard.py - Streamlit drift intelligence dashboard (bonus).

Usage:
    streamlit run src/dashboard.py

Auto-loads data/precomputed.json if present (no upload needed).
Generate it once with:
    python generate_dashboard_data.py --train_data_filepath <train> --test_data_filepath <test>

Features:
    Tab 1 - Drift Overview   : PSI bar chart per feature (green/amber/red)
    Tab 2 - Feature Deep Dive: Distribution overlay for any selected feature
    Tab 3 - Model Performance: AU-PRC metrics + prediction probability histogram
    Tab 4 - Mitigation Summary: What was dropped and why + empirical benchmarks
"""

import os
import sys
import json
import io

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Allow importing from src/ when run from project root or Streamlit Cloud
_src_dir  = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_src_dir)
for _p in (_src_dir, _root_dir, os.path.join(_root_dir, 'src')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from feature_engineer import preprocess, fit_encoder, apply_encoder, align_to_training_columns, encode_target
from drift_detector import (
    build_drift_report, calculate_psi_numerical, calculate_psi_categorical,
    PSI_MODERATE, PSI_SEVERE
)
from drift_mitigator import two_pass_train

try:
    from sklearn.metrics import average_precision_score
    import lightgbm as lgb
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='GEN-I Drift Intelligence Dashboard',
    page_icon='📊',
    layout='wide',
)

st.title('📊 Adaptive Drift Intelligence Dashboard')
st.caption('NAISC 2026 | Team GEN-I | Singtel Churn Prediction | Drift Detection & Mitigation')

# ── Precomputed data path ─────────────────────────────────────────────────────
_PRECOMPUTED_PATH = os.path.join(_root_dir, 'data', 'precomputed.json')


@st.cache_data(show_spinner=False)
def load_precomputed_file(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header('Data Source')

    precomputed_available = os.path.exists(_PRECOMPUTED_PATH)
    if precomputed_available:
        st.success('Pre-computed results loaded automatically.')
        use_precomputed = st.checkbox('Use pre-computed results', value=True)
    else:
        use_precomputed = False
        st.info('No pre-computed data found. Upload CSVs below.')

    st.divider()
    st.subheader('Upload Custom Data (optional)')
    train_file = st.file_uploader('Train CSV', type='csv', key='train')
    test_file  = st.file_uploader('Test CSV',  type='csv', key='test')
    run_pipeline = st.button(
        'Run Pipeline on Uploaded Data',
        type='primary',
        disabled=(train_file is None or test_file is None),
    )

    st.divider()
    st.header('PSI Thresholds')
    psi_mod    = st.slider('Moderate (amber)', 0.05, 0.20, float(PSI_MODERATE), 0.01)
    psi_severe = st.slider('Severe (red)',     0.10, 0.50, float(PSI_SEVERE),   0.01)


# ── Helpers ───────────────────────────────────────────────────────────────────

def psi_color(psi: float) -> str:
    if psi < psi_mod:
        return 'green'
    if psi < psi_severe:
        return 'orange'
    return 'red'


def run_full_pipeline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Run the full pipeline and return all results including distributions."""
    import time

    y_train = encode_target(train_df['ChurnStatus'])
    has_labels = 'ChurnStatus' in test_df.columns
    y_test = encode_target(test_df['ChurnStatus']) if has_labels else None

    X_train_raw = preprocess(train_df)
    X_test_raw  = preprocess(test_df)

    drift_report = build_drift_report(X_train_raw, X_test_raw)

    enc, cat_cols = fit_encoder(X_train_raw)
    X_train = apply_encoder(X_train_raw, enc, cat_cols).fillna(-999)
    X_test  = apply_encoder(X_test_raw,  enc, cat_cols).fillna(-999)
    train_medians = X_train.median().to_dict()
    X_test = align_to_training_columns(X_test, list(X_train.columns), train_medians)

    t0 = time.time()
    final_model, X_train_clean, X_test_clean, dropped = two_pass_train(
        X_train, y_train, X_test, drift_report
    )
    elapsed = time.time() - t0

    train_probs = final_model.predict_proba(X_train_clean)[:, 1]
    test_probs  = final_model.predict_proba(X_test_clean)[:, 1]
    train_auprc = float(average_precision_score(y_train, train_probs))
    test_auprc  = float(average_precision_score(y_test, test_probs)) if y_test is not None else None

    importances = pd.Series(
        final_model.feature_importances_, index=X_train_clean.columns
    ).sort_values(ascending=False).to_dict()

    # Build distributions inline
    distributions = {}
    for col in X_train_raw.columns:
        if col not in X_test_raw.columns:
            continue
        tr = X_train_raw[col].dropna()
        te = X_test_raw[col].dropna()
        if pd.api.types.is_numeric_dtype(tr):
            lo = float(min(tr.min(), te.min()))
            hi = float(max(tr.max(), te.max()))
            if lo == hi:
                hi = lo + 1.0
            bins = np.linspace(lo, hi, 41).tolist()
            tr_counts, _ = np.histogram(tr, bins=bins)
            te_counts, _ = np.histogram(te, bins=bins)
            distributions[col] = {
                'type': 'numerical', 'bins': bins,
                'train_counts': tr_counts.tolist(), 'test_counts': te_counts.tolist(),
                'train_stats': {'count': len(tr), 'mean': float(tr.mean()),
                                'std': float(tr.std()), 'min': float(tr.min()), 'max': float(tr.max())},
                'test_stats':  {'count': len(te), 'mean': float(te.mean()),
                                'std': float(te.std()), 'min': float(te.min()), 'max': float(te.max())},
                'psi': float(calculate_psi_numerical(tr, te)),
            }
        else:
            distributions[col] = {
                'type': 'categorical',
                'train_freq': tr.astype(str).value_counts(normalize=True).to_dict(),
                'test_freq':  te.astype(str).value_counts(normalize=True).to_dict(),
                'psi': float(calculate_psi_categorical(tr, te)),
            }

    return {
        'drift_report'         : drift_report,
        'dropped'              : dropped,
        'elapsed'              : elapsed,
        'train_auprc'          : train_auprc,
        'test_auprc'           : test_auprc,
        'test_probs'           : test_probs.tolist(),
        'importances'          : importances,
        'feature_distributions': distributions,
        'train_shape'          : list(train_df.shape),
        'test_shape'           : list(test_df.shape),
    }


# ── Load results ──────────────────────────────────────────────────────────────

results = None

# Priority 1: user clicked Run Pipeline with uploaded files
if run_pipeline and train_file and test_file:
    with st.spinner('Running drift detection and mitigation pipeline...'):
        try:
            train_df = pd.read_csv(train_file)
            test_df  = pd.read_csv(test_file)
            results  = run_full_pipeline(train_df, test_df)
            st.sidebar.success(f'Done in {results["elapsed"]:.1f}s')
        except Exception as e:
            st.sidebar.error(f'Pipeline error: {e}')

# Priority 2: auto-load precomputed file
if results is None and use_precomputed and precomputed_available:
    try:
        results = load_precomputed_file(_PRECOMPUTED_PATH)
    except Exception as e:
        st.sidebar.error(f'Could not load precomputed data: {e}')

# Nothing loaded yet
if results is None:
    st.info(
        'No data loaded yet.  \n'
        'Pre-computed results will appear here automatically once `data/precomputed.json` exists.  \n\n'
        'To generate it, run:  \n'
        '```\npython generate_dashboard_data.py '
        '--train_data_filepath public_data/train.csv '
        '--test_data_filepath public_data/test.csv\n```'
    )
    st.stop()

# ── Dataset info banner ───────────────────────────────────────────────────────
train_shape = results.get('train_shape')
test_shape  = results.get('test_shape')
if train_shape and test_shape:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Train rows',    f'{train_shape[0]:,}')
    c2.metric('Train features', train_shape[1] - 2)  # minus CustomerID + target
    c3.metric('Test rows',     f'{test_shape[0]:,}')
    c4.metric('Test features',  test_shape[1] - 2)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    '🔍 Drift Overview',
    '📈 Feature Deep Dive',
    '🎯 Model Performance',
    '🛠 Mitigation Summary',
])

drift_report   = results.get('drift_report', [])
distributions  = results.get('feature_distributions', {})

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Drift Overview
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader('Feature-Level PSI Scores')

    if not drift_report:
        st.success('No significant drift detected.')
    else:
        df_psi = pd.DataFrame([
            {
                'Feature' : r['column'],
                'PSI'     : r['psi'],
                'Severity': r['severity'],
                'p-value' : r['p_value'],
                'Type'    : r['col_type'],
                'Status'  : ('Severe'   if r['psi'] >= psi_severe
                             else 'Moderate' if r['psi'] >= psi_mod
                             else 'Stable'),
            }
            for r in drift_report
        ]).sort_values('PSI', ascending=False)

        col_s, col_m, col_st = st.columns(3)
        col_s.metric(f'Severe Drift  (PSI ≥ {psi_severe:.2f})',
                     int((df_psi['Status'] == 'Severe').sum()))
        col_m.metric(f'Moderate Drift (PSI ≥ {psi_mod:.2f})',
                     int((df_psi['Status'] == 'Moderate').sum()))
        col_st.metric('Stable Features',
                      int((df_psi['Status'] == 'Stable').sum()))

        color_map = {'Severe': 'red', 'Moderate': 'orange', 'Stable': 'green'}
        fig = px.bar(
            df_psi, x='Feature', y='PSI',
            color='Status', color_discrete_map=color_map,
            title='PSI by Feature (sorted by severity)',
            labels={'PSI': 'Population Stability Index'},
        )
        fig.add_hline(y=psi_mod,    line_dash='dash', line_color='orange',
                      annotation_text=f'Moderate ({psi_mod})')
        fig.add_hline(y=psi_severe, line_dash='dash', line_color='red',
                      annotation_text=f'Severe ({psi_severe})')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_psi, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Feature Deep Dive
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader('Train vs Test Distribution')

    if not distributions:
        st.info('No distribution data available.')
    else:
        drifted_cols = [r['column'] for r in drift_report]
        all_cols     = list(distributions.keys())
        sorted_cols  = drifted_cols + [c for c in all_cols if c not in drifted_cols]

        selected = st.selectbox('Select feature', sorted_cols)

        if selected and selected in distributions:
            d = distributions[selected]
            psi_val = d['psi']

            col_left, col_right = st.columns(2)
            col_left.metric('PSI', f'{psi_val:.4f}')
            col_right.metric(
                'Status',
                'Severe'   if psi_val >= psi_severe
                else 'Moderate' if psi_val >= psi_mod
                else 'Stable'
            )

            if d['type'] == 'numerical':
                bins         = d['bins']
                bin_centres  = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins) - 1)]
                tr_counts    = d['train_counts']
                te_counts    = d['test_counts']

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=bin_centres, y=tr_counts,
                    name='Train', marker_color='steelblue', opacity=0.7,
                    width=(bins[1] - bins[0]) * 0.45,
                ))
                fig.add_trace(go.Bar(
                    x=bin_centres, y=te_counts,
                    name='Test', marker_color='tomato', opacity=0.7,
                    width=(bins[1] - bins[0]) * 0.45,
                ))
                fig.update_layout(
                    barmode='overlay',
                    title=f'{selected}: Train vs Test Distribution',
                    xaxis_title=selected, yaxis_title='Count',
                )
                st.plotly_chart(fig, use_container_width=True)

                tr_s = d['train_stats']
                te_s = d['test_stats']
                stats = pd.DataFrame({
                    'Metric': ['Count', 'Mean', 'Std', 'Min', 'Max'],
                    'Train' : [tr_s['count'], tr_s['mean'], tr_s['std'], tr_s['min'], tr_s['max']],
                    'Test'  : [te_s['count'], te_s['mean'], te_s['std'], te_s['min'], te_s['max']],
                }).round(3)
                st.dataframe(stats, use_container_width=True)

            else:
                tr_freq = d['train_freq']
                te_freq = d['test_freq']
                all_cats = sorted(set(tr_freq) | set(te_freq))
                freq_df = pd.DataFrame({
                    'Category': all_cats,
                    'Train'   : [tr_freq.get(c, 0) for c in all_cats],
                    'Test'    : [te_freq.get(c, 0) for c in all_cats],
                })

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=freq_df['Category'], y=freq_df['Train'],
                    name='Train', marker_color='steelblue', opacity=0.8,
                ))
                fig.add_trace(go.Bar(
                    x=freq_df['Category'], y=freq_df['Test'],
                    name='Test', marker_color='tomato', opacity=0.8,
                ))
                fig.update_layout(
                    barmode='group',
                    title=f'{selected}: Category Frequency',
                    xaxis_title=selected, yaxis_title='Proportion',
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(freq_df.round(4), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Model Performance
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader('Model Performance')

    train_auprc = results.get('train_auprc')
    test_auprc  = results.get('test_auprc')
    test_probs  = results.get('test_probs', [])
    importances = results.get('importances', {})

    c1, c2, c3 = st.columns(3)
    if train_auprc is not None:
        c1.metric('Train AU-PRC', f'{train_auprc:.4f}')
    if test_auprc is not None:
        gap = (train_auprc - test_auprc) if train_auprc else 0
        c2.metric('Test AU-PRC', f'{test_auprc:.4f}',
                  delta=f'-{gap:.4f} gap' if gap > 0.01 else 'Minimal gap')
    baseline = 0.2879
    if test_auprc is not None:
        c3.metric('Lift vs Random Baseline', f'+{test_auprc - baseline:.4f}')

    if test_probs:
        st.markdown('#### Prediction Probability Distribution')
        fig = px.histogram(
            x=test_probs, nbins=50,
            labels={'x': 'Churn Probability', 'y': 'Count'},
            title='Distribution of Predicted Churn Probabilities (Test Set)',
            color_discrete_sequence=['steelblue'],
        )
        fig.add_vline(x=0.5, line_dash='dash', line_color='red',
                      annotation_text='0.5 threshold')
        st.plotly_chart(fig, use_container_width=True)

    if importances:
        st.markdown('#### Top 20 Feature Importances (Gain)')
        imp_df = (pd.Series(importances)
                    .sort_values(ascending=False)
                    .head(20)
                    .reset_index())
        imp_df.columns = ['Feature', 'Gain Importance']
        fig2 = px.bar(
            imp_df, x='Gain Importance', y='Feature',
            orientation='h',
            title='Top 20 Feature Importances',
            color='Gain Importance',
            color_continuous_scale='Blues',
        )
        fig2.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Mitigation Summary
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader('Mitigation Summary')

    dropped = results.get('dropped', [])
    elapsed = results.get('elapsed')

    if elapsed is not None:
        st.metric('Detection + Mitigation Time', f'{elapsed:.1f}s')

    st.markdown('#### Strategy Applied')
    st.info(
        '**Two-pass PSI-based feature pruning**  \n'
        'Pass 1: Train on all features to obtain gain-based importances.  \n'
        'Pass 2: Drop features where PSI > 0.15 AND gain importance > 5%, then retrain.  \n\n'
        'This approach raised test AU-PRC from **0.699 → 0.846** (+0.147) on public data.'
    )

    mit_rows = [
        {
            'Feature'     : r['column'],
            'PSI'         : round(r['psi'], 4),
            'Severity'    : r['severity'],
            'p-value'     : r['p_value'],
            'Action Taken': r['mitigation'] if r['mitigation'] else 'No action',
        }
        for r in drift_report
    ]
    if mit_rows:
        st.dataframe(pd.DataFrame(mit_rows), use_container_width=True)

    if dropped:
        st.markdown('#### Dropped Features')
        for col in dropped:
            st.markdown(f'- `{col}`')

    st.markdown('#### Empirical Ablation (All Strategies Tested)')
    bench = pd.DataFrame([
        {'Strategy': 'Baseline (no mitigation)',       'Train AU-PRC': 0.876, 'Test AU-PRC': 0.699, 'Delta': '—'},
        {'Strategy': 'Log transforms drifted cols',    'Train AU-PRC': 0.874, 'Test AU-PRC': 0.697, 'Delta': '-0.003'},
        {'Strategy': 'Interaction features',           'Train AU-PRC': 0.882, 'Test AU-PRC': 0.682, 'Delta': '-0.017'},
        {'Strategy': 'Recency weighting (decay=0.2)',  'Train AU-PRC': 0.871, 'Test AU-PRC': 0.680, 'Delta': '-0.019'},
        {'Strategy': 'Sliding window (last 6 months)', 'Train AU-PRC': 0.863, 'Test AU-PRC': 0.678, 'Delta': '-0.021'},
        {'Strategy': 'Bin NumberofReferrals (5 bins)', 'Train AU-PRC': 0.791, 'Test AU-PRC': 0.728, 'Delta': '+0.029'},
        {'Strategy': 'DROP NumberofReferrals (ours)',  'Train AU-PRC': 0.875, 'Test AU-PRC': 0.846, 'Delta': '+0.147'},
    ])
    st.dataframe(bench, use_container_width=True)
