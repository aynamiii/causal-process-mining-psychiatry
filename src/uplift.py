from __future__ import annotations
import numpy as np
import pandas as pd
import math
from causalml.inference.tree import UpliftTreeClassifier
from config import UPLIFT_MAX_DEPTH, UPLIFT_MIN_LEAF_PCT

CONFOUNDER_COLS = ['age_at_admission']


def _prepare_confounder_matrix(cases_df: pd.DataFrame, condition_cols: list | None = None) -> np.ndarray:
    df = cases_df.copy()
    for col in CONFOUNDER_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['is_female'] = (df['gender'].astype(str) == '8532').astype(int)
    df['is_male']   = (df['gender'].astype(str) == '8507').astype(int)
    cols = ['is_female', 'is_male'] + [c for c in CONFOUNDER_COLS if c in df.columns]
    if condition_cols:
        for c in condition_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        cols += [c for c in condition_cols if c in df.columns]
    X = df[cols].fillna(df[cols].median()).values
    return X


def _leaf_ite_per_patient(model: UpliftTreeClassifier, X: np.ndarray) -> np.ndarray:

    tree = model.fitted_uplift_tree

    def _score(node, x):
        if node.col == -1:
            return float(node.upliftScore[0]) if node.upliftScore else 0.0
        if x[node.col] >= node.value:
            return _score(node.trueBranch, x)
        return _score(node.falseBranch, x)

    return np.array([_score(tree, X[i]) for i in range(len(X))])


def _wilson_or_normal_diff_ci(p1, n1, p2, n2, z=1.96):

    if n1 == 0 or n2 == 0:
        return None, None
    se = math.sqrt((p1 * (1 - p1)) / n1 + (p2 * (1 - p2)) / n2)
    diff = p1 - p2
    return round(diff - z * se, 4), round(diff + z * se, 4)


def _minimum_detectable_effect(n1, n2, baseline_p=0.5, alpha=0.05, power=0.80):

    if n1 == 0 or n2 == 0:
        return None
    z_alpha = 1.96
    z_beta = 0.8416
    var_term = baseline_p * (1 - baseline_p) * (1 / n1 + 1 / n2)
    return round((z_alpha + z_beta) * math.sqrt(var_term), 4)


def describe_tree(model, feature_names: list, baseline_rate: float = None) -> list[dict]:

    rows = []

    def _walk(node, depth, path: list[str]):
        if node is None:
            return
        is_leaf = node.col == -1

        n_samples = None
        control_rate = control_n = treatment_rate = treatment_n = None
        ite_recomputed = ci_lower = ci_upper = mde = None

        if node.nodeSummary:
            control_rate, control_n = node.nodeSummary[0]
            treatment_rate, treatment_n = node.nodeSummary[1]
            n_samples = control_n + treatment_n

            ite_recomputed = round(treatment_rate - control_rate, 4)
            ci_lower, ci_upper = _wilson_or_normal_diff_ci(
                treatment_rate, treatment_n, control_rate, control_n
            )
            pooled_baseline = baseline_rate if baseline_rate is not None else control_rate
            mde = _minimum_detectable_effect(treatment_n, control_n, baseline_p=pooled_baseline)

        ite = round(float(node.upliftScore[0]), 4) if node.upliftScore else None

        rows.append({
            'depth':            depth,
            'node_type':        'leaf' if is_leaf else 'split',
            'split_on':         '—' if is_leaf else feature_names[node.col],
            'cut_point':        '—' if is_leaf else round(float(node.value), 2),
            'path':             ' & '.join(path) if path else '(all patients)',
            'n_samples':        n_samples,
            'ite':              ite,
            'ite_recomputed':   ite_recomputed,   # cross-check against 'ite'
            'control_n':        control_n,
            'control_rate':     round(control_rate, 4) if control_rate is not None else None,
            'treatment_n':      treatment_n,
            'treatment_rate':   round(treatment_rate, 4) if treatment_rate is not None else None,
            'ci_95_lower':      ci_lower,
            'ci_95_upper':      ci_upper,
            'crosses_zero':     (ci_lower is not None and ci_lower < 0 < ci_upper),
            'min_detectable_effect_80pct_power': mde,
        })
        if not is_leaf:
            feat = feature_names[node.col]
            cp   = round(float(node.value), 2)
            _walk(node.trueBranch,  depth + 1, path + [f'{feat} ≥ {cp}'])
            _walk(node.falseBranch, depth + 1, path + [f'{feat} < {cp}'])

    _walk(model.fitted_uplift_tree, depth=0, path=[])
    return rows


def fit_uplift(
    cases_df: pd.DataFrame,
    treatments: list,
    condition_cols: list | None = None,
) -> tuple[pd.DataFrame, dict, dict]:
    X = _prepare_confounder_matrix(cases_df, condition_cols)
    y = (1 - cases_df['readmitted'].astype(int).values)
    min_leaf = max(10, int(len(cases_df) * UPLIFT_MIN_LEAF_PCT))

    feature_names = ['is_female', 'is_male'] + [c for c in CONFOUNDER_COLS if c in cases_df.columns]
    if condition_cols:
        feature_names += [c for c in condition_cols if c in cases_df.columns]

    results = []
    fitted_models = {}
    ite_arrays = {}
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
            random_state=42,
        )
        m.fit(X, treatment=np.where(t_vec == 1, 'treatment', 'control'), y=y)

        ite = _leaf_ite_per_patient(m, X)
        ite_arrays[col] = ite

        results.append({
            'treatment': col,
            'n_treated': n_t,
            'max_ite':   round(float(np.max(ite)), 4),
            'mean_ite':  round(float(np.mean(ite)), 4),
        })
        fitted_models[col] = (m, feature_names)

    if not results:
        return pd.DataFrame(columns=['treatment', 'n_treated', 'max_ite', 'mean_ite']), {}, {}
    summary = pd.DataFrame(results).sort_values('max_ite', ascending=False).reset_index(drop=True)
    return summary, fitted_models, ite_arrays
