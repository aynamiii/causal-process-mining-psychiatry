import pandas as pd
from config import READMISSION_DAYS


def label_readmission(
    cases_df: pd.DataFrame,
    visit_df: pd.DataFrame,
    window: int = READMISSION_DAYS,
) -> pd.DataFrame:

    v = visit_df[['visit_occurrence_id', 'person_id',
                  'visit_start_date', 'visit_end_date']].copy()
    v['visit_start_date'] = pd.to_datetime(v['visit_start_date'])
    v['visit_end_date'] = pd.to_datetime(v['visit_end_date'])
    v = v.sort_values(['person_id', 'visit_start_date'])

    v['next_visit'] = v.groupby('person_id')['visit_start_date'].shift(-1)
    v['days_to_next'] = (v['next_visit'] - v['visit_end_date']).dt.days
    v['readmitted'] = (
        (v['days_to_next'] >= 0) & (v['days_to_next'] <= window)
    ).astype(int)

    cases_df = cases_df.merge(
        v[['visit_occurrence_id', 'readmitted']],
        on='visit_occurrence_id', how='left'
    )
    cases_df['readmitted'] = cases_df['readmitted'].fillna(0).astype(int)
    return cases_df
