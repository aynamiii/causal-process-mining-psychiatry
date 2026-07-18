from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from causalml.inference.tree import UpliftTreeClassifier
from config import UPLIFT_MAX_DEPTH, UPLIFT_MIN_LEAF_PCT

CONFOUNDER_COLS = ['age_at_admission', 'length_of_stay']


def _prepare_confounder_matrix(cases_df: pd.DataFrame, condition_cols: list | None = None) -> np.ndarray:
    df = cases_df.copy()
    for col in CONFOUNDER_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    le = LabelEncoder().fit(df['gender'].fillna('U').astype(str).unique())
    df['gender_enc'] = le.transform(df['gender'].fillna('U').astype(str))
    cols = ['gender_enc'] + [c for c in CONFOUNDER_COLS if c in df.columns]
    if condition_cols:
        for c in condition_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        cols += [c for c in condition_cols if c in df.columns]
    X = df[cols].fillna(df[cols].median()).values
    return X


def describe_tree(model: UpliftTreeClassifier, feature_names: list) -> list[dict]:
    """Walk a fitted uplift tree and return one dict per node."""
    rows = []

    def _walk(node, depth):
        if node is None:
            return
        is_leaf = node.col == -1
        # nodeSummary: [[rate, n], ...] — first entry is control, second is treatment
        n_samples = sum(grp[1] for grp in node.nodeSummary) if node.nodeSummary else None
        ite = round(float(node.upliftScore[0]), 4) if node.upliftScore else None
        rows.append({
            'depth':      depth,
            'node_type':  'leaf' if is_leaf else 'split',
            'split_on':   '—' if is_leaf else feature_names[node.col],
            'n_samples':  n_samples,
            'ite':        ite,
        })
        if not is_leaf:
            _walk(node.trueBranch,  depth + 1)
            _walk(node.falseBranch, depth + 1)

    _walk(model.fitted_uplift_tree, depth=0)
    return rows


def fit_uplift(cases_df: pd.DataFrame, treatments: list, condition_cols: list | None = None) -> tuple[pd.DataFrame, dict]:
    X = _prepare_confounder_matrix(cases_df, condition_cols)
    y = (1 - cases_df['readmitted'].astype(int).values)
    min_leaf = max(10, int(len(cases_df) * UPLIFT_MIN_LEAF_PCT))

    # feature names mirror _prepare_confounder_matrix column order
    feature_names = ['gender_enc'] + [c for c in CONFOUNDER_COLS if c in cases_df.columns]
    if condition_cols:
        feature_names += [c for c in condition_cols if c in cases_df.columns]

    results = []
    fitted_models = {}
    for col in treatments:
        t_vec = pd.to_numeric(cases_df[col], errors='coerce').fillna(0).astype(int).values
        n_t = int(t_vec.sum())
        if n_t < 10 or int((t_vec == 0).sum()) < 10:
            continue

        m = UpliftTreeClassifier(
            control_name='control',
            max_depth=UPLIFT_MAX_DEPTH,
            min_samples_leaf=min_leaf,
            evaluationFunction='KL',
            honesty=True,
            estimation_sample_size=0.5,
        )
        m.fit(X, treatment=np.where(t_vec == 1, 'treatment', 'control'), y=y)
        preds = m.predict(X)
        ite = preds[:, 0]
        results.append({
            'treatment': col,
            'n_treated': n_t,
            'max_ite': round(float(np.max(ite)), 4),
        })
        fitted_models[col] = (m, feature_names)

    if not results:
        return pd.DataFrame(columns=['treatment', 'n_treated', 'max_ite']), {}
    summary = pd.DataFrame(results).sort_values('max_ite', ascending=False).reset_index(drop=True)
    return summary, fitted_models
