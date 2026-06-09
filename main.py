import os
import matplotlib.pyplot as plt

import config
from src.data_loading import load_all_tables
from src.cohort import build_cohort, filter_visits_to_cohort
from src.feature_engineering import build_features
from src.outcome_labelling import label_readmission
from src.action_rules import prepare_ar_dataframe, run_action_rules, extract_treatments
from src.uplift import fit_uplift
from src.causal_rules import assemble_causal_rules, evaluate_rules

os.makedirs(config.OUTPUT_DIR, exist_ok=True)


def main():
    # 1. Load data
    print('1.Loading OMOP tables')
    tables = load_all_tables(config.DATA_DIR)

    # 2. Build cohort
    print('2.Building depression cohort')
    cohort_df = build_cohort(
        condition_df = tables['condition_occurrence'],
        visit_df = tables['visit_occurrence'],
        drug_df = tables['drug_exposure'],
        concept_ancestor_df = tables['concept_ancestor'],
    )
    print(f'Final cohort: {len(cohort_df):,} patients')

    cohort_visits = filter_visits_to_cohort(tables['visit_occurrence'], cohort_df)
    print(f'Cohort visits (post-diagnosis): {len(cohort_visits):,}')

    # 3. Build features
    print('3.Building features ')
    cases_df, drug_cols, therapy_cols = build_features(
        visit_df = cohort_visits,
        person_df = tables['person'],
        condition_df = tables['condition_occurrence'],
        drug_df = tables['drug_exposure'],
        procedure_df = tables['procedure_occurrence'],
        concept_df = tables['concept'],
        concept_ancestor_df = tables['concept_ancestor'],
    )
    print(f'Cases: {len(cases_df):,} | '
          f'Flexible: {len(drug_cols + therapy_cols)} attributes')

    # 4. Label outcome
    print('4.Labelling readmission')
    cases_df = label_readmission(cases_df, cohort_visits)
    print(f'Readmission rate ({config.READMISSION_DAYS}-day): '
          f'{cases_df["readmitted"].mean():.1%}')

    # 5. Action rule mining
    print('5.Action rule mining')
    ar_df, stable_cols, flexible_cols = prepare_ar_dataframe(
        cases_df, drug_cols, therapy_cols
    )
    _, pretty_rules, json_export = run_action_rules(
        ar_df, stable_cols, flexible_cols
    )
    with open(os.path.join(config.OUTPUT_DIR, 'action_rules.txt'), 'w') as f:
        f.write('\n'.join(pretty_rules))

    # 6. Extract treatment candidates
    print('6.Extracting treatment candidates')
    top_treatments = extract_treatments(json_export, flexible_cols)
    if not top_treatments:
        top_treatments = (cases_df[flexible_cols].sum()
                          .sort_values(ascending=False).head(5).index.tolist())
    print(f'Candidates: {top_treatments}')

    # 7. Uplift trees
    print('7.Uplift modelling')
    uplift_summary = fit_uplift(cases_df, top_treatments)
    uplift_summary.to_csv(
        os.path.join(config.OUTPUT_DIR, 'uplift_summary.csv'), index=False
    )

    # 8. Assemble causal rules
    print('8.Assembling causal rules ')
    causal_rules_df = assemble_causal_rules(
        json_export, uplift_summary, flexible_cols
    )
    print(f'Causal rules (ATE > 0): {len(causal_rules_df)}')
    for _, row in causal_rules_df.iterrows():
        print(' •', row['rule'])

    if len(causal_rules_df) > 0:
        causal_rules_df[['treatment', 'precondition', 'change', 'ATE', 'n_treated', 'rule']
            ].to_csv(os.path.join(config.OUTPUT_DIR, 'causal_rules.csv'), index=False)

    # 9. Evaluate
    print('9.Evaluating rules')
    eval_df = evaluate_rules(cases_df, uplift_summary)
    eval_df.to_csv(os.path.join(config.OUTPUT_DIR, 'rule_evaluation.csv'), index=False)
    print(eval_df.to_string(index=False))

    # Plot uplift
    if len(uplift_summary) > 0:
        fig, ax = plt.subplots(figsize=(9, max(3, len(uplift_summary) * 0.55)))
        colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in uplift_summary['ATE']]
        ax.barh(uplift_summary['treatment'], uplift_summary['ATE'],
                color=colors, edgecolor='white', height=0.6)
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xlabel('ATE  (positive = reduces readmission)')
        ax.set_title('Causal effect of treatments on 30-day readmission\n(Depression cohort)')
        plt.tight_layout()
        plt.savefig(os.path.join(config.OUTPUT_DIR, 'uplift_scores.png'), dpi=150)
        print(f'\nPlot saved to {config.OUTPUT_DIR}/uplift_scores.png')

    print('\n── Done ──')
    print(f'Outputs saved to: {config.OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
