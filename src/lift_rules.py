import json
import itertools
import pandas as pd
from config import AR_MIN_SUPPORT_PCT, AR_MIN_CONFIDENCE, AR_MIN_STABLE, AR_MAX_STABLE, AR_MIN_FLEXIBLE


def _stable_subgroups(ar_df: pd.DataFrame, stable_cols: list, min_stable: int):

    for r in range(min_stable, min(AR_MAX_STABLE, len(stable_cols)) + 1):
        for combo in itertools.combinations(stable_cols, r):
            for keys, grp in ar_df.groupby(list(combo)):
                if not isinstance(keys, tuple):
                    keys = (keys,)
                yield dict(zip(combo, keys)), grp


def run_lift_rules(
    ar_df: pd.DataFrame,
    stable_cols: list,
    flexible_cols: list,
) -> tuple:

    n_total = len(ar_df)
    min_support = max(1, int(n_total * AR_MIN_SUPPORT_PCT))
    target = 'readmitted_str'

    flex_cols_present = [
        c for c in flexible_cols
        if c in ar_df.columns and 'no_matching_concept' not in c
    ]

    rules = []
    pretty = []

    for stable_dict, grp in _stable_subgroups(ar_df, stable_cols, AR_MIN_STABLE):
        grp_size = len(grp)
        if grp_size < min_support:
            continue

        undesired = grp[grp[target] == '1']
        desired = grp[grp[target] == '0']

        if len(undesired) < min_support:
            continue

        flex_hits = []
        for col in flex_cols_present:
            col_vals = pd.to_numeric(grp[col], errors='coerce').fillna(0)
            col_undesired = pd.to_numeric(undesired[col], errors='coerce').fillna(0)
            col_desired = pd.to_numeric(desired[col], errors='coerce').fillna(0)

            u_support = int((col_undesired == 0).sum())
            if u_support < min_support:
                continue

            d_support = int((col_desired == 1).sum())
            if d_support < min_support:
                continue

            col_one_count = int((col_vals == 1).sum())
            if col_one_count == 0:
                continue
            confidence = d_support / col_one_count
            if confidence < AR_MIN_CONFIDENCE:
                continue

            flex_hits.append({
                'attribute': col,
                'from': '0',
                'to': '1',
                'candidates': u_support,
                'evidence': d_support,
                'confidence': round(confidence, 4),
            })

        if len(flex_hits) < AR_MIN_FLEXIBLE:
            continue

        _GENDER = {'8507': 'Male', '8532': 'Female'}
        stable_str = ', '.join(
            f'{k}={_GENDER.get(str(v), v) if "gender" in k else v}'
            for k, v in stable_dict.items()
        )
        for hit in flex_hits:
            rule = {
                'stable': [{'attribute': k, 'value': v} for k, v in stable_dict.items()],
                'flexible': [hit],
                'candidates': hit['candidates'],
                'evidence': hit['evidence'],
                'confidence': hit['confidence'],
            }
            rules.append(rule)
            col_name = hit['attribute']
            pretty.append(
                f"[{stable_str}] ({col_name}: 0→1) "
                f"[candidates: {hit['candidates']}, evidence: {hit['evidence']}, "
                f"conf: {hit['confidence']:.2f}]"
            )

    return pretty, rules


def prepare_ar_dataframe(
    cases_df: pd.DataFrame,
    stable_cols: list,
    flexible_cols: list,
) -> pd.DataFrame:
    df = cases_df.copy()
    df['readmitted_str'] = df['readmitted'].astype(str)
    keep = [c for c in stable_cols + flexible_cols if c in df.columns]
    return df[keep + ['readmitted_str']].reset_index(drop=True)


def extract_treatments(json_export: object, flexible_cols: list) -> list:
    flex_set = set(flexible_cols)
    counts = {}
    rules = json_export if isinstance(json_export, list) else json.loads(json_export)
    for rule in (rules if isinstance(rules, list) else []):
        for item in rule.get('flexible', []):
            a = item.get('attribute', '')
            if a in flex_set:
                counts[a] = counts.get(a, 0) + 1
    return [t for t, _ in sorted(counts.items(), key=lambda x: -x[1])]
