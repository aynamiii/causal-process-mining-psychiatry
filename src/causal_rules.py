import json
import pandas as pd


def assemble_causal_rules(
    json_export: object,
    uplift_summary: pd.DataFrame,
    flexible_cols: list,
    min_ate: float = 0.0,
) -> pd.DataFrame:

    flex_set = set(flexible_cols)
    rows = []

    rules = json.loads(json_export) if isinstance(json_export, str) else json_export
    for rule in (rules if isinstance(rules, list) else []):
        flex = rule.get('flexible', rule.get('flexible_attributes', {}))
        stable = rule.get('stable', rule.get('stable_attributes', {}))

        treatments = (
            [(a, f"{v[0]}→{v[1]}" if isinstance(v, list)
              else f"{v.get('from','0')}→{v.get('to','1')}")
             for a, v in flex.items() if a in flex_set]
            if isinstance(flex, dict) else
            [(i.get('attribute', ''),
              f"{i.get('from','0')}→{i.get('to','1')}")
             for i in flex if i.get('attribute', '') in flex_set]
        )
        precond = (
            ', '.join(f'{a}={v}' for a, v in stable.items())
            if isinstance(stable, dict) else
            ', '.join(f"{i['attribute']}={i['value']}"
                      for i in stable)
        ) or 'any patient'

        for attr, change in treatments:
            rows.append({'treatment': attr, 'change': change,
                         'precondition': precond})

    causal_df = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=['treatment', 'precondition'])
        .merge(uplift_summary[['treatment', 'ATE', 'n_treated', 'n_control']],
               on='treatment', how='inner')
    )
    causal_df = causal_df[causal_df['ATE'] > min_ate].sort_values('ATE', ascending=False)

    causal_df['rule'] = causal_df.apply(
        lambda r: (
            f"IF [{r['precondition']}]"
            f"THEN [{r['treatment']}: {r['change']}]"
            f"→ readmission↓  "
            f"[ATE={r['ATE']:+.4f}, n={r['n_treated']}]"
        ), axis=1
    )
    return causal_df


def evaluate_rules(
    cases_df: pd.DataFrame,
    uplift_summary: pd.DataFrame,
) -> pd.DataFrame:

    rows = []
    for _, r in uplift_summary.iterrows():
        col = r['treatment']
        if col not in cases_df.columns:
            continue
        treated = cases_df[cases_df[col] == 1]['readmitted']
        untreated = cases_df[cases_df[col] == 0]['readmitted']
        rows.append({
            'treatment': col,
            'readmit_treated': round(treated.mean(), 3),
            'readmit_control': round(untreated.mean(),3),
            'absolute_reduction': round(untreated.mean() - treated.mean(), 3),
            'ATE': r['ATE'],
            'n_treated': len(treated),
        })
    return pd.DataFrame(rows)
