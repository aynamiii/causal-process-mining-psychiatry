import os
import pandas as pd
import config
from src.preprocessing import prepare_data
from src.lift_rules import prepare_ar_dataframe


_GENDER_REVERSE = {'Female': '8532', 'Male': '8507'}
_AGE_REVERSE = {
    'under 40': 'age_lt40', '40-54': 'age_40_54', '55-64': 'age_55_64',
    '65-74': 'age_65_74', '75 and over': 'age_gte75',
}
_LOS_REVERSE = {
    'same day': 'same_day', '1-3 days': 'short',
    '4-7 days': 'medium', 'over 7 days': 'long',
}


def _display_name_to_column(display_name: str, stable_cols: list) -> str:
    key = display_name.lower().replace(' ', '_')
    if key in ('gender', 'age_group', 'los_category'):
        return key
    candidate = f'cond__{key}'
    if candidate in stable_cols:
        return candidate
    if key in stable_cols:
        return key
    raise ValueError(f"Could not map display name '{display_name}' to a known column")


def _parse_precondition(precondition: str, stable_cols: list) -> dict:
    if precondition.strip() == 'all patients':
        return {}
    cleaned = precondition.replace('patients with ', '')
    result = {}
    for pair in cleaned.split(', '):
        display_attr, display_val = pair.split(' = ')
        display_attr = display_attr.strip()
        display_val = display_val.strip()
        col = _display_name_to_column(display_attr, stable_cols)
        if col == 'gender':
            val = _GENDER_REVERSE.get(display_val, display_val)
        elif col == 'age_group':
            val = _AGE_REVERSE.get(display_val, display_val)
        elif col == 'los_category':
            val = _LOS_REVERSE.get(display_val, display_val)
        else:
            val = display_val
        result[col] = val
    return result


def compute_lift_for_subgroup(ar_df, stable_dict, flex_col):
    mask = pd.Series(True, index=ar_df.index)
    for k, v in stable_dict.items():
        mask &= (ar_df[k].astype(str) == str(v))
    grp = ar_df[mask]
    if len(grp) == 0:
        return None

    col_vals = pd.to_numeric(grp[flex_col], errors='coerce').fillna(0)
    treated = grp[col_vals == 1]
    untreated = grp[col_vals == 0]
    if len(treated) == 0 or len(untreated) == 0:
        return None

    conf_treated = (treated['readmitted_str'] == '0').mean()
    conf_untreated = (untreated['readmitted_str'] == '0').mean()
    lift = conf_treated / conf_untreated if conf_untreated > 0 else float('inf')

    return {
        'n_treated': len(treated),
        'n_untreated': len(untreated),
        'confidence_treated': round(conf_treated, 4),
        'confidence_untreated': round(conf_untreated, 4),
        'lift': round(lift, 4),
    }


def main():
    print('Rebuilding cases_df / ar_df (reusing cached event log if present)...')
    cases_df, stable_cols, flexible_cols, _ = prepare_data(config.DATA_DIR)
    ar_df = prepare_ar_dataframe(cases_df, stable_cols, flexible_cols)
    print(f'  ar_df: {len(ar_df)} rows')

    causal_path = os.path.join(config.OUTPUT_DIR, 'causal_rules.csv')
    causal_df = pd.read_csv(causal_path)
    print(f'Loaded {len(causal_df)} rules from {causal_path}')

    results = []
    for _, row in causal_df.iterrows():
        flex_col = row['treatment']
        precondition = row['precondition']
        try:
            stable_dict = _parse_precondition(precondition, stable_cols)
        except ValueError as e:
            print(f'  SKIP (could not parse): {precondition} -- {e}')
            continue

        lift_result = compute_lift_for_subgroup(ar_df, stable_dict, flex_col)
        if lift_result is None:
            print(f'  SKIP (empty group): {precondition} / {flex_col}')
            continue

        results.append({
            'treatment': flex_col,
            'precondition': precondition,
            'subgroup_ite_from_causal_rules': row['subgroup_ite'],
            **lift_result,
        })

    out_df = pd.DataFrame(results)
    out_path = os.path.join(config.OUTPUT_DIR, 'causal_rules_with_lift.csv')
    out_df.to_csv(out_path, index=False)

    print(f'\n{len(out_df)} rules processed. Saved to {out_path}\n')
    print(out_df.to_string())


if __name__ == '__main__':
    main()
