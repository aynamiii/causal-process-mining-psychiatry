import json
import pandas as pd
from action_rules import ActionRules
from config import (AR_MIN_STABLE, AR_MIN_FLEXIBLE,
                    AR_MIN_SUPPORT, AR_MIN_CONFIDENCE)


def prepare_ar_dataframe(
    cases_df: pd.DataFrame,
    drug_cols: list,
    therapy_cols: list,
) -> tuple[pd.DataFrame, list, list]:

    df = cases_df.copy()

    df['age_group'] = pd.cut(
        df['age'], bins=[0, 40, 55, 65, 75, 150],
        labels=['age_lt40', 'age_40_54', 'age_55_64', 'age_65_74', 'age_gte75']
    ).astype(str)

    df['cci_group'] = pd.cut(
        df['cci_score'], bins=[-1, 0, 2, 4, 100],
        labels=['none', 'mild', 'moderate', 'severe']
    ).astype(str)

    df['los_category'] = pd.cut(
        df['los_days'], bins=[-1, 0, 3, 7, 100],
        labels=['same_day', 'short', 'medium', 'long']
    ).astype(str)

    stable_cols = ['age_group', 'cci_group', 'los_category', 'gender', 'visit_type']
    flexible_cols = drug_cols + therapy_cols

    for c in flexible_cols:
        df[c] = df[c].astype(str)

    df['readmitted_str'] = df['readmitted'].astype(str)

    ar_df = df[stable_cols + flexible_cols + ['readmitted_str']].reset_index(drop=True)
    return ar_df, stable_cols, flexible_cols


def run_action_rules(
    ar_df: pd.DataFrame,
    stable_cols: list,
    flexible_cols: list,
) -> tuple:

    model = ActionRules(
        min_stable_attributes = AR_MIN_STABLE,
        min_flexible_attributes = AR_MIN_FLEXIBLE,
        min_undesired_support = AR_MIN_SUPPORT,
        min_undesired_confidence = AR_MIN_CONFIDENCE,
        min_desired_support = AR_MIN_SUPPORT,
        min_desired_confidence = AR_MIN_CONFIDENCE,
        verbose = True,
    )
    model.fit(
        data = ar_df,
        stable_attributes = stable_cols,
        flexible_attributes = flexible_cols,
        target = 'readmitted_str',
        target_undesired_state = '1',
        target_desired_state = '0',
        use_sparse_matrix = False,
    )
    rules_obj = model.get_rules()
    pretty_rules = rules_obj.get_pretty_ar_notation()
    json_export = rules_obj.get_export_notation()

    return model, pretty_rules, json_export


def extract_treatments(
    json_export: object,
    flexible_cols: list,
    top_n: int = 5,
) -> list:

    flex_set = set(flexible_cols)
    counts = {}

    rules = json.loads(json_export) if isinstance(json_export, str) else json_export
    for rule in (rules if isinstance(rules, list) else []):
        flex = rule.get('flexible', rule.get('flexible_attributes', {}))
        attrs = flex.keys() if isinstance(flex, dict) else [
            i.get('attribute', '') for i in flex
        ]
        for a in attrs:
            if a in flex_set:
                counts[a] = counts.get(a, 0) + 1

    return [t for t, _ in sorted(counts.items(), key=lambda x: -x[1])][:top_n]
