import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from causalml.inference.tree import UpliftTreeClassifier
from config import UPLIFT_MAX_DEPTH, UPLIFT_MIN_LEAF_PCT

CONFOUNDER_COLS = ['age_at_admission', 'length_of_stay']


def prepare_confounder_matrix(cases_df: pd.DataFrame) -> np.ndarray:
    df = cases_df.copy()
    for col in CONFOUNDER_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    le = LabelEncoder().fit(df['gender'].fillna('U').astype(str).unique())
    df['gender_enc'] = le.transform(df['gender'].fillna('U').astype(str))
    cols = ['gender_enc'] + [c for c in CONFOUNDER_COLS if c in df.columns]
    X = df[cols].fillna(df[cols].median()).values
    return X


def fit_uplift(cases_df: pd.DataFrame, treatments: list) -> pd.DataFrame:
    X = prepare_confounder_matrix(cases_df)
    y = (1 - cases_df['readmitted'].astype(int).values)
    min_leaf = max(10, int(len(cases_df) * UPLIFT_MIN_LEAF_PCT))

    results = []
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
            honesty=False,
        )
        m.fit(X, treatment=np.where(t_vec == 1, 'treatment', 'control'), y=y)
        ite = m.predict(X)[:, 0] - m.predict(X)[:, 1]
        results.append({
            'treatment': col,
            'n_treated': n_t,
            'max_ite': round(float(np.max(ite)), 4),
        })

    if not results:
        return pd.DataFrame(columns=['treatment', 'n_treated', 'max_ite'])
    return pd.DataFrame(results).sort_values('max_ite', ascending=False).reset_index(drop=True)
