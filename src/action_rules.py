import json
import pandas as pd
from action_rules import ActionRules
from config import AR_MIN_STABLE, AR_MIN_FLEXIBLE, AR_MIN_SUPPORT_PCT, AR_MIN_CONFIDENCE, AR_VERBOSE


def prepare_ar_dataframe(
    cases_df: pd.DataFrame,
    stable_cols: list,
    flexible_cols: list,
) -> pd.DataFrame:
    df = cases_df.copy()
    df['readmitted_str'] = df['readmitted'].astype(str)
    keep = [c for c in stable_cols + flexible_cols if c in df.columns]
    return df[keep + ['readmitted_str']].reset_index(drop=True)


def run_action_rules(
    ar_df: pd.DataFrame,
    stable_cols: list,
    flexible_cols: list,
) -> tuple:
    min_support = max(1, int(len(ar_df) * AR_MIN_SUPPORT_PCT))
    model = ActionRules(
        min_stable_attributes=AR_MIN_STABLE,
        min_flexible_attributes=AR_MIN_FLEXIBLE,
        min_undesired_support=min_support,
        min_undesired_confidence=AR_MIN_CONFIDENCE,
        min_desired_support=min_support,
        min_desired_confidence=AR_MIN_CONFIDENCE,
        verbose=AR_VERBOSE,
    )
    model.fit(
        data=ar_df,
        stable_attributes=stable_cols,
        flexible_attributes=flexible_cols,
        target='readmitted_str',
        target_undesired_state='1',
        target_desired_state='0',
        use_sparse_matrix=False,
    )
    rules_obj = model.get_rules()
    return rules_obj.get_pretty_ar_notation(), rules_obj.get_export_notation()


def extract_treatments(json_export: object, flexible_cols: list) -> list:
    flex_set = set(flexible_cols)
    counts = {}
    rules = json.loads(json_export) if isinstance(json_export, str) else json_export
    for rule in (rules if isinstance(rules, list) else []):
        flex = rule.get('flexible', rule.get('flexible_attributes', {}))
        attrs = flex.keys() if isinstance(flex, dict) else [i.get('attribute', '') for i in flex]
        for a in attrs:
            if a in flex_set:
                counts[a] = counts.get(a, 0) + 1
    return [t for t, _ in sorted(counts.items(), key=lambda x: -x[1])]
