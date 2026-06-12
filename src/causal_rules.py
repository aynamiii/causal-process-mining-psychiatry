import json
import pandas as pd
from config import UPLIFT_MIN_MAX_ITE


_AGE_LABELS = {
    'age_lt40':   'under 40',
    'age_40_54':  '40–54',
    'age_55_64':  '55–64',
    'age_65_74':  '65–74',
    'age_gte75':  '75 and over',
}

_LOS_LABELS = {
    'same_day': 'same day',
    'short':    '1–3 days',
    'medium':   '4–7 days',
    'long':     'over 7 days',
}

def _clean(name: str) -> str:
    for prefix in ('act__', 'cond__'):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name.replace('_', ' ').title()


def _clean_value(attr: str, val: str) -> str:
    if 'age_group' in attr:
        return _AGE_LABELS.get(val, val)
    if 'los_category' in attr:
        return _LOS_LABELS.get(val, val)
    return val


def _fmt_stable(stable: object) -> list[tuple]:
    if isinstance(stable, dict):
        return list(stable.items())
    return [(i['attribute'], i['value']) for i in stable]


def _action_phrase(treatment: str, frm: str, to: str) -> str:
    name = _clean(treatment)
    if frm.strip() == '0' and to.strip() == '1':
        return f'Administering {name}'
    if frm.strip() == '1' and to.strip() == '0':
        return f'Discontinuing {name}'
    return f'Changing {name} from {frm} to {to}'


def _subpopulation_phrase(stable_items: list[tuple]) -> str:
    if not stable_items:
        return 'all patients'
    parts = []
    for attr, val in stable_items:
        parts.append(f'{_clean(attr)} = {_clean_value(attr, val)}')
    return 'patients with ' + ', '.join(parts)


def assemble_causal_rules(
    json_export: object,
    uplift_summary: pd.DataFrame,
    flexible_cols: list,
    min_max_ite: float = UPLIFT_MIN_MAX_ITE,
) -> pd.DataFrame:
    flex_set = set(flexible_cols)
    rows = []

    rules = json.loads(json_export) if isinstance(json_export, str) else json_export
    for rule in (rules if isinstance(rules, list) else []):
        flex = rule.get('flexible', rule.get('flexible_attributes', {}))
        stable = rule.get('stable', rule.get('stable_attributes', {}))

        treatments = (
            [(a, f"{v[0]}→{v[1]}" if isinstance(v, list) else f"{v.get('from','0')}→{v.get('to','1')}")
             for a, v in flex.items() if a in flex_set]
            if isinstance(flex, dict) else
            [(i.get('attribute', ''), f"{i.get('from','0')}→{i.get('to','1')}")
             for i in flex if i.get('attribute', '') in flex_set]
        )

        stable_items = _fmt_stable(stable)
        for attr, change in treatments:
            frm, to = change.split('→')
            rows.append({
                'treatment': attr,
                'change': change,
                'precondition': _subpopulation_phrase(stable_items),
                'n_stable': len(stable_items),
                'action': _action_phrase(attr, frm, to),
            })

    if not rows:
        return pd.DataFrame(columns=['treatment', 'precondition', 'change', 'max_ite', 'n_treated', 'rule'])

    causal_df = (
        pd.DataFrame(rows)
        .merge(uplift_summary[['treatment', 'max_ite', 'n_treated']], on='treatment', how='inner')
    )
    causal_df = causal_df[causal_df['max_ite'] > min_max_ite]

    causal_df = (
        causal_df.sort_values(['max_ite', 'n_stable'], ascending=[False, True])
        .drop_duplicates(subset=['treatment'])
        .drop(columns=['n_stable'])
        .reset_index(drop=True)
    )

    causal_df['rule'] = [
        (
            f"Action {i + 1}: {row['action']}. "
            f"Sub-population: {row['precondition']}. "
            f"[support: {row['n_treated']}, max_ite: {row['max_ite']:+.4f}]"
        )
        for i, row in causal_df.iterrows()
    ]
    return causal_df
