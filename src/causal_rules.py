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

_GENDER_LABELS = {
    '8507': 'Male',
    '8532': 'Female',
}

def _clean(name: str) -> str:
    for prefix in ('act__', 'cond__', 'drug__', 'proc__'):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name.replace('_', ' ').title()


def _clean_value(attr: str, val: str) -> str:
    if 'age_group' in attr:
        return _AGE_LABELS.get(val, val)
    if 'los_category' in attr:
        return _LOS_LABELS.get(val, val)
    if 'gender' in attr:
        return _GENDER_LABELS.get(str(val), val)
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

        stable_items = _fmt_stable(stable)
        if stable_items and all(str(v) == '0' for _, v in stable_items):
            continue
        if isinstance(flex, dict):
            # dict format: {attr: [from, to]} or {attr: {'from':..,'to':..}}
            flex_items = [
                {'attribute': a,
                 'from': str(v[0]) if isinstance(v, list) else str(v.get('from', '0')),
                 'to':   str(v[1]) if isinstance(v, list) else str(v.get('to', '1')),
                 'candidates': 0}
                for a, v in flex.items()
            ]
        else:
            flex_items = flex
        for item in flex_items:
            attr       = item.get('attribute', '')
            frm        = item.get('from', '0')
            to         = item.get('to', '1')
            candidates = item.get('candidates', item.get('undesired_support', 0))
            if attr not in flex_set:
                continue
            rows.append({
                'treatment': attr,
                'change': f'{frm}→{to}',
                'precondition': _subpopulation_phrase(stable_items),
                'n_stable': len(stable_items),
                'action': _action_phrase(attr, frm, to),
                '_stable_items': stable_items,
                'n_candidates': candidates,
            })

    if not rows:
        return pd.DataFrame(columns=['treatment', 'precondition', 'change', 'max_ite', 'n_candidates', 'rule'])

    causal_df = (
        pd.DataFrame(rows)
        .merge(uplift_summary[['treatment', 'max_ite']], on='treatment', how='inner')
    )
    causal_df = causal_df[causal_df['max_ite'] > min_max_ite]

    causal_df['positive_precondition'] = causal_df['_stable_items'].apply(
        lambda items: _subpopulation_phrase([(a, v) for a, v in items if str(v) != '0'])
    )
    causal_df = (
        causal_df.sort_values(['max_ite', 'n_stable'], ascending=[False, True])
        .drop_duplicates(subset=['treatment', 'positive_precondition'])
        .drop(columns=['n_stable', 'positive_precondition', '_stable_items'])
        .reset_index(drop=True)
    )

    causal_df['rule'] = [
        (
            f"Action {i + 1}: {row['action']}. "
            f"Sub-population: {row['precondition']}. "
            f"[candidates: {row['n_candidates']}, max_ite: {row['max_ite']:+.4f}]"
        )
        for i, row in causal_df.iterrows()
    ]
    return causal_df.drop(columns=['action'])
