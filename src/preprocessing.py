from __future__ import annotations
import os
import re
import pandas as pd
from airms_connect.connection import airms_connection
from config import ACTIVITY_MIN_FREQUENCY, TOP_N_DRUGS, TOP_N_PROCEDURES

STABLE_EVENT_TYPES = {'CONDITION'}


def _sanitize(name: str) -> str:
    name = re.sub(r'[^a-z0-9]+', '_', str(name).lower().strip())
    return name.strip('_')[:60]


def _onehot_activities(event_log, event_types, prefix, case_col, min_count, top_n=None):
    subset = event_log[event_log['event_type'].str.upper().isin(event_types)]
    if subset.empty:
        return pd.DataFrame()
    pivot = subset.groupby([case_col, 'activity']).size().unstack(fill_value=0).clip(upper=1)
    freq = pivot.sum().sort_values(ascending=False)
    if top_n is not None:
        pivot = pivot[freq[freq >= min_count].head(top_n).index]
    else:
        pivot = pivot[freq[freq >= min_count].index]
    pivot.columns = [f'{prefix}__{_sanitize(c)}' for c in pivot.columns]
    pivot.columns.name = None
    return pivot.astype(int)


_EVENT_LOG_SQL = """
SELECT
    e.PERSON_ID,
    e.INDEX_VISIT_OCCURRENCE_ID,
    e.ACTIVITY,
    e.EVENT_DATETIME,
    e.EVENT_TYPE,
    YEAR(c.admission_date) - p.YEAR_OF_BIRTH        AS age_at_admission,
    p.GENDER_CONCEPT_NAME                            AS gender,
    DAYS_BETWEEN(c.admission_date, c.discharge_date) AS length_of_stay,
    COALESCE(rf.readmitted, 0)                       AS readmitted
FROM soleiz01.RCA_log e
INNER JOIN soleiz01.depression_inpatient_cohort c
    ON c.index_visit_occurrence_id = e.INDEX_VISIT_OCCURRENCE_ID
INNER JOIN CDMPHI.PERSON p
    ON p.PERSON_ID = e.PERSON_ID
LEFT JOIN soleiz01.readmission_flag rf
    ON rf.index_visit_occurrence_id = e.INDEX_VISIT_OCCURRENCE_ID
WHERE e.INDEX_VISIT_OCCURRENCE_ID IN (SELECT case_id FROM soleiz01.case_sample)
ORDER BY e.INDEX_VISIT_OCCURRENCE_ID, e.EVENT_DATETIME
"""


def _fetch_event_log(data_dir: str) -> pd.DataFrame:
    cache_path = os.path.join(data_dir, 'event_log_sample.csv')
    if os.path.exists(cache_path):
        print(f"  Loading cached event log from {cache_path}")
        return pd.read_csv(cache_path, parse_dates=['event_datetime'], low_memory=False)

    print("  Connecting to AIRMS/CDMPHI...")
    airms = airms_connection()
    airms.on_minerva(login_host_name="li04e01")
    airms.connect()

    event_log = airms.conn.sql(_EVENT_LOG_SQL).collect()
    event_log.columns = event_log.columns.str.lower()

    os.makedirs(data_dir, exist_ok=True)
    event_log.to_csv(cache_path, index=False)
    print(f"  Saved event log to {cache_path}")
    return event_log


def prepare_data(data_dir: str, min_frequency: float = ACTIVITY_MIN_FREQUENCY) -> tuple[pd.DataFrame, list, list]:
    event_log = _fetch_event_log(data_dir)
    event_log.columns = event_log.columns.str.lower()
    event_log['event_datetime'] = pd.to_datetime(event_log['event_datetime'], errors='coerce')

    case_col = 'index_visit_occurrence_id'
    min_count = max(1, int(event_log[case_col].nunique() * min_frequency))

    drug_df = _onehot_activities(event_log, {'DRUG'}, 'drug', case_col, min_count, top_n=TOP_N_DRUGS)
    proc_df = _onehot_activities(event_log, {'PROCEDURE'}, 'proc', case_col, min_count, top_n=TOP_N_PROCEDURES)
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
    ).cat.add_categories('unknown').fillna('unknown').astype(str)

    case_attrs['los_category'] = pd.cut(
        pd.to_numeric(case_attrs['length_of_stay'], errors='coerce').clip(lower=0),
        bins=[-1, 0, 3, 7, 9999],
        labels=['same_day', 'short', 'medium', 'long'],
    ).cat.add_categories('unknown').fillna('unknown').astype(str)

    df = case_attrs.copy()
    for ohe_df in [drug_df, proc_df, stable_act_df]:
        if not ohe_df.empty:
            ohe_df.index = ohe_df.index.astype(df.index.dtype)
            df = df.join(ohe_df, how='left')

    act_cols = list(drug_df.columns) + list(proc_df.columns) + list(stable_act_df.columns)
    df[act_cols] = df[act_cols].fillna(0).astype(int)
    df = df.reset_index(drop=True)

    stable_cols = ['age_group', 'gender', 'los_category'] + list(stable_act_df.columns)
    flexible_cols = list(drug_df.columns) + list(proc_df.columns)

    for c in stable_cols + flexible_cols:
        if c in df.columns:
            df[c] = df[c].astype(str)

    return df, stable_cols, flexible_cols
