import os

import config
from src.preprocessing import prepare_data
from src.action_rules import prepare_ar_dataframe, run_action_rules, extract_treatments
from src.uplift import fit_uplift
from src.causal_rules import assemble_causal_rules

os.makedirs(config.OUTPUT_DIR, exist_ok=True)


def main():
    print('1. Preparing data')
    cases_df, stable_cols, flexible_cols = prepare_data(config.DATA_DIR)

    print('2. Action rule mining')
    ar_df = prepare_ar_dataframe(cases_df, stable_cols, flexible_cols)
    pretty_rules, json_export = run_action_rules(ar_df, stable_cols, flexible_cols)
    with open(os.path.join(config.OUTPUT_DIR, 'action_rules.txt'), 'w') as f:
        f.write('\n'.join(pretty_rules))
    print(f'   Rules found: {len(pretty_rules)}')

    print('3. Extracting treatments')
    treatments = extract_treatments(json_export, flexible_cols) or flexible_cols
    print(f'   Treatments: {len(treatments)}')

    print('4. Uplift modelling')
    uplift_summary = fit_uplift(cases_df, treatments)

    print('5. Assembling causal rules')
    causal_rules_df = assemble_causal_rules(json_export, uplift_summary, flexible_cols)
    print(f'   Causal rules found: {len(causal_rules_df)}')
    for _, row in causal_rules_df.iterrows():
        print(' •', row['rule'])

    causal_rules_df[['treatment', 'precondition', 'change', 'max_ite', 'n_treated', 'rule']].to_csv(
        os.path.join(config.OUTPUT_DIR, 'causal_rules.csv'), index=False
    )
    print(f'\nDone. Output saved to {config.OUTPUT_DIR}/causal_rules.csv')


if __name__ == '__main__':
    main()
