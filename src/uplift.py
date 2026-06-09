import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from causalml.inference.tree import UpliftTreeClassifier
from config import UPLIFT_MAX_DEPTH, UPLIFT_MIN_LEAF


def prepare_confounder_matrix(cases_df: pd.DataFrame) -> tuple[np.ndarray, list]:

    df = cases_df.copy()
    le_gender = LabelEncoder().fit(df['gender'].fillna('U').unique())
    le_visittype = LabelEncoder().fit(df['visit_type'].fillna('other').unique())
    df['gender_enc'] = le_gender.transform(df['gender'].fillna('U'))
    df['visit_type_enc'] = le_visittype.transform(df['visit_type'].fillna('other'))

    feature_names = ['age', 'gender_enc', 'visit_type_enc', 'cci_score', 'los_days']
    X = df[feature_names].fillna(df[feature_names].median()).values
    return X, feature_names


def fit_uplift(
    cases_df: pd.DataFrame,
    top_treatments: list,
) -> pd.DataFrame:

    X, _ = prepare_confounder_matrix(cases_df)
    y = (1 - cases_df['readmitted'].values)

    results = []
    for col in top_treatments:
        t_vec = cases_df[col].values
        n_t = int(t_vec.sum())
        n_c = int((t_vec == 0).sum())

        if n_t < 10 or n_c < 10:
            continue

        m = UpliftTreeClassifier(
            control_name = 'control',
            max_depth = UPLIFT_MAX_DEPTH,
            min_samples_leaf = UPLIFT_MIN_LEAF,
            evaluationFunction = 'KL',
            honesty = False,
        )
        m.fit(X, treatment=np.where(t_vec == 1, 'treatment', 'control'), y=y)
        s = m.predict(X)
        ate = float(np.mean(s[:, 0] - s[:, 1]))

        results.append({
            'treatment': col,
            'n_treated': n_t,
            'n_control': n_c,
            'ATE': round(ate, 4),
        })

    return pd.DataFrame(results).sort_values('ATE', ascending=False)
