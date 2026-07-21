import os

import pandas as pd
import config
from src.preprocessing import prepare_data
from src.lift_rules import prepare_ar_dataframe, run_lift_rules as run_action_rules, extract_treatments
from src.uplift import fit_uplift, describe_tree
from src.eda import save_eda
from src.causal_rules import assemble_causal_rules

os.makedirs(config.OUTPUT_DIR, exist_ok=True)


def main():
    print('1. Preparing data')
    cases_df, stable_cols, flexible_cols, event_log = prepare_data(config.DATA_DIR)

    print('   EDA summary')
    save_eda(event_log, config.OUTPUT_DIR)

    print('2. Action rule mining')
    ar_df = prepare_ar_dataframe(cases_df, stable_cols, flexible_cols)
    pretty_rules, json_export = run_action_rules(ar_df, stable_cols, flexible_cols)
    with open(os.path.join(config.OUTPUT_DIR, 'action_rules.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(pretty_rules))
    print(f'   Rules found: {len(pretty_rules)}')

    print('3. Extracting treatments')
    treatments = extract_treatments(json_export, flexible_cols) or flexible_cols
    print(f'   Treatments: {len(treatments)} drugs')

    print('4. Uplift modelling')
    condition_cols = [c for c in stable_cols if c.startswith('cond__')]
    uplift_summary, fitted_models, ite_arrays = fit_uplift(cases_df, treatments, condition_cols)

    print('5. Assembling causal rules')
    causal_rules_df = assemble_causal_rules(json_export, uplift_summary, flexible_cols, cases_df, ite_arrays)
    print(f'   Causal rules found: {len(causal_rules_df)}')
    for _, row in causal_rules_df.iterrows():
        print(' •', row['rule'])

    causal_rules_df[['treatment', 'precondition', 'change', 'subgroup_ite', 'n_candidates', 'rule']].to_csv(
        os.path.join(config.OUTPUT_DIR, 'causal_rules.csv'), index=False
    )

    print('6. Saving tree breakdowns for all fitted treatments')
    tree_rows = []
    for treatment, (model, feature_names) in fitted_models.items():
        nodes = describe_tree(model, feature_names)
        for node in nodes:
            tree_rows.append({'treatment': treatment, **node})

    if tree_rows:
        tree_df = pd.DataFrame(tree_rows)
        tree_df.to_csv(os.path.join(config.OUTPUT_DIR, 'tree_breakdowns.csv'), index=False)

        print(f'   Saved {len(tree_rows)} nodes for {len(fitted_models)} treatment(s)')

    print(f'\nDone. Outputs saved to {config.OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
