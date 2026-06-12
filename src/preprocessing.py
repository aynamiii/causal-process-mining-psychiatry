import os
import re
import pandas as pd
from config import ACTIVITY_MIN_FREQUENCY

FLEXIBLE_EVENT_TYPES = {'DRUG', 'PROCEDURE'}
STABLE_EVENT_TYPES = {'CONDITION'}


def _sanitize(name: str) -> str:
    name = re.sub(r'[^a-z0-9]+', '_', str(name).lower().strip())
    return name.strip('_')[:60]


def _onehot_activities(event_log, event_types, prefix, case_col, min_count):
    subset = event_log[event_log['event_type'].str.upper().isin(event_types)]
    if subset.empty:
        return pd.DataFrame()
    pivot = subset.groupby([case_col, 'activity']).size().unstack(fill_value=0).clip(upper=1)
    freq = pivot.sum()
    pivot = pivot[freq[freq >= min_count].index]
    pivot.columns = [f'{prefix}__{_sanitize(c)}' for c in pivot.columns]
    pivot.columns.name = None
    return pivot.astype(int)


def prepare_data(data_dir: str, min_frequency: float = ACTIVITY_MIN_FREQUENCY) -> tuple[pd.DataFrame, list, list]:
    event_log = pd.read_csv(os.path.join(data_dir, 'event_log_sample.csv'), parse_dates=['event_datetime'], low_memory=False)
    event_log.columns = event_log.columns.str.lower()

    case_col = 'index_visit_occurrence_id'
    min_count = max(1, int(event_log[case_col].nunique() * min_frequency))

    flex_df = _onehot_activities(event_log, FLEXIBLE_EVENT_TYPES, 'act', case_col, min_count)
    stable_act_df = _onehot_activities(event_log, STABLE_EVENT_TYPES, 'cond', case_col, min_count)

    case_attrs = (
        event_log.drop_duplicates(subset=[case_col])
        [[case_col, 'age_at_admission', 'gender', 'length_of_stay', 'readmitted']]
        .set_index(case_col)
    )

    case_attrs['age_group'] = pd.cut(
        pd.to_numeric(case_attrs['age_at_admission'], errors='coerce'),
        bins=[0, 40, 55, 65, 75, 150],
        labels=['age_lt40', 'age_40_54', 'age_55_64', 'age_65_74', 'age_gte75'],
    ).astype(str)

    case_attrs['los_category'] = pd.cut(
        pd.to_numeric(case_attrs['length_of_stay'], errors='coerce').clip(lower=0),
        bins=[-1, 0, 3, 7, 9999],
        labels=['same_day', 'short', 'medium', 'long'],
    ).astype(str)

    df = case_attrs.copy()
    for ohe_df in [flex_df, stable_act_df]:
        if not ohe_df.empty:
            ohe_df.index = ohe_df.index.astype(df.index.dtype)
            df = df.join(ohe_df, how='left')

    act_cols = list(flex_df.columns) + list(stable_act_df.columns)
    df[act_cols] = df[act_cols].fillna(0).astype(int)
    df = df.reset_index(drop=True)

    stable_cols = ['age_group', 'gender', 'los_category'] + list(stable_act_df.columns)
    flexible_cols = list(flex_df.columns)

    for c in stable_cols + flexible_cols:
        if c in df.columns:
            df[c] = df[c].astype(str)

    return df, stable_cols, flexible_cols
