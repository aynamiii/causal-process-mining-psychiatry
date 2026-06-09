import pandas as pd
from config import (
    DEPRESSION_SNOMED_CONCEPTS,
    ANTIDEPRESSANT_CONCEPTS,
    COHORT_LOOKBACK_DAYS,
    TREATMENT_NAIVE_GRACE_DAYS,
)
from src.utils import get_descendant_ids


def _all_antidepressant_ancestor_ids() -> list:
    ids = []
    for class_ids in ANTIDEPRESSANT_CONCEPTS.values():
        ids.extend(class_ids)
    return list(set(ids))


def build_cohort(
    condition_df: pd.DataFrame,
    visit_df: pd.DataFrame,
    drug_df: pd.DataFrame,
    concept_ancestor_df: pd.DataFrame,
) -> pd.DataFrame:

    # Expand depression concepts to all descendants
    depression_concept_ids = get_descendant_ids(
        concept_ancestor_df, DEPRESSION_SNOMED_CONCEPTS
    )

    # Find patients with a depression diagnosis
    cond = condition_df[['person_id', 'condition_concept_id', 'condition_start_date']].copy()
    cond['condition_start_date'] = pd.to_datetime(cond['condition_start_date'])

    depression_cond = cond[cond['condition_concept_id'].isin(depression_concept_ids)]

    # Earliest depression diagnosis per patient
    first_diagnosis = (
        depression_cond.groupby('person_id')['condition_start_date']
        .min()
        .reset_index()
        .rename(columns={'condition_start_date': 'first_depression_diagnosis'})
    )

    # Find first record per patient (visit or condition)
    visit_df = visit_df.copy()
    condition_df = condition_df.copy()
    visit_df['visit_start_date'] = pd.to_datetime(visit_df['visit_start_date'])
    condition_df['condition_start_date'] = pd.to_datetime(condition_df['condition_start_date'])

    first_visit = (
        visit_df.groupby('person_id')['visit_start_date']
        .min().reset_index()
        .rename(columns={'visit_start_date': 'first_visit_date'})
    )
    first_condition = (
        condition_df.groupby('person_id')['condition_start_date']
        .min().reset_index()
        .rename(columns={'condition_start_date': 'first_condition_date'})
    )

    # Take the earliest of the two as the patient's first record
    first_records = first_visit.merge(first_condition, on='person_id', how='outer')
    first_records['first_record_date'] = first_records[
        ['first_visit_date', 'first_condition_date']
    ].min(axis=1)

    # Apply 365-day lookback
    cohort = first_diagnosis.merge(
        first_records[['person_id', 'first_record_date']],
        on='person_id', how='left'
    )
    cohort['days_before_diagnosis'] = (
        cohort['first_depression_diagnosis'] - cohort['first_record_date']
    ).dt.days

    cohort = cohort[cohort['days_before_diagnosis'] >= COHORT_LOOKBACK_DAYS].copy()
    print(f'  After {COHORT_LOOKBACK_DAYS}-day lookback filter: {len(cohort):,} patients')

    # Find first antidepressant exposure per patient
    print('  Expanding antidepressant concepts via concept_ancestor...')
    ad_ancestor_ids = _all_antidepressant_ancestor_ids()
    ad_concept_ids = get_descendant_ids(concept_ancestor_df, ad_ancestor_ids)
    print(f'  Antidepressant concept IDs (inc. descendants): {len(ad_concept_ids):,}')

    drug_df = drug_df.copy()
    drug_df['drug_exposure_start_date'] = pd.to_datetime(drug_df['drug_exposure_start_date'])

    ad_exposures = drug_df[drug_df['drug_concept_id'].isin(ad_concept_ids)].copy()

    first_ad = (
        ad_exposures.groupby('person_id')['drug_exposure_start_date']
        .min().reset_index()
        .rename(columns={'drug_exposure_start_date': 'first_ad_exposure'})
    )

    cohort = cohort.merge(first_ad, on='person_id', how='left')

    # Treatment-naive filter
    cohort['days_ad_before_diagnosis'] = (
        cohort['first_depression_diagnosis'] - cohort['first_ad_exposure']
    ).dt.days.fillna(0)

    cohort = cohort[
        cohort['first_ad_exposure'].isna() |
        (cohort['days_ad_before_diagnosis'] <= TREATMENT_NAIVE_GRACE_DAYS)
    ].copy()

    print(f'  After treatment-naive filter: {len(cohort):,} patients')
    print(f'  Of which received antidepressants: '
          f'{cohort["first_ad_exposure"].notna().sum():,}')

    return cohort.reset_index(drop=True)


def filter_visits_to_cohort(
    visit_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
) -> pd.DataFrame:

    visit_df = visit_df.copy()
    visit_df['visit_start_date'] = pd.to_datetime(visit_df['visit_start_date'])

    # Merge in each patient's first diagnosis date
    visit_df = visit_df.merge(
        cohort_df[['person_id', 'first_depression_diagnosis']],
        on='person_id', how='inner'
    )

    # Keep visits on or after first diagnosis
    visit_df = visit_df[
        visit_df['visit_start_date'] >= visit_df['first_depression_diagnosis']
    ].drop(columns=['first_depression_diagnosis'])

    return visit_df.reset_index(drop=True)
