from __future__ import annotations
import os
import pandas as pd
from config import TOP_N_DRUGS, TOP_N_PROCEDURES, ACTIVITY_MIN_FREQUENCY


def save_eda(event_log: pd.DataFrame, output_dir: str) -> None:
    el = event_log.copy()
    el.columns = el.columns.str.lower()

    visit_cols = ['index_visit_occurrence_id',
                  'age_at_admission', 'gender', 'length_of_stay', 'readmitted']
    if 'person_id' in el.columns:
        visit_cols = ['person_id'] + visit_cols

    visits = (
        el.drop_duplicates('index_visit_occurrence_id')
        [visit_cols]
        .reset_index(drop=True)
    )

    n_visits   = visits['index_visit_occurrence_id'].nunique()
    n_patients = visits['person_id'].nunique() if 'person_id' in visits.columns else None
    min_count  = max(1, int(n_visits * ACTIVITY_MIN_FREQUENCY))

    lines = []
    w = lines.append

    # ── Cohort & Outcome ─────────────────────────────────────────────────────
    n_readmitted  = int(visits['readmitted'].sum())
    readmit_rate  = visits['readmitted'].mean() * 100

    w('=' * 60)
    w('COHORT & OUTCOME')
    w('=' * 60)
    w(f'  Index admissions (cases) : {n_visits:,}')
    if n_patients is not None:
        w(f'  Unique patients          : {n_patients:,}')
    w(f'  Readmitted within 30d    : {n_readmitted:,}  ({readmit_rate:.1f}%)')
    w('')

    # ── Demographics & Stay ───────────────────────────────────────────────────
    gender     = visits['gender'].value_counts(dropna=False)
    gender_str = '  |  '.join(
        f'{g}: {n:,} ({n / n_visits * 100:.1f}%)' for g, n in gender.items()
    )

    age = pd.to_numeric(visits['age_at_admission'], errors='coerce').dropna()
    age_str = (
        f'median {age.median():.0f} yrs  '
        f'(mean {age.mean():.0f},  '
        f'IQR {age.quantile(0.25):.0f}–{age.quantile(0.75):.0f},  '
        f'range {age.min():.0f}–{age.max():.0f})'
    )

    los = pd.to_numeric(visits['length_of_stay'], errors='coerce').dropna()
    los_str = (
        f'median {los.median():.0f} days  '
        f'(mean {los.mean():.1f},  '
        f'IQR {los.quantile(0.25):.0f}–{los.quantile(0.75):.0f},  '
        f'range {los.min():.0f}–{los.max():.0f})'
    )

    w('DEMOGRAPHICS & STAY')
    w('=' * 60)
    w(f'  Gender split   : {gender_str}')
    w(f'  Age            : {age_str}')
    w(f'  Length of stay : {los_str}')
    w('')

    # ── Event Log ─────────────────────────────────────────────────────────────
    event_counts = el['event_type'].str.upper().value_counts()

    w('EVENT LOG')
    w('=' * 60)
    w(f'  Total events        : {len(el):,}')
    w(f'  Drug events         : {event_counts.get("DRUG", 0):,}')
    w(f'  Procedure events    : {event_counts.get("PROCEDURE", 0):,}')
    w(f'  Condition events    : {event_counts.get("CONDITION", 0):,}')
    w(f'  Avg events/visit    : {len(el) / n_visits:.1f}')
    w('')

    # ── Pipeline features ────────────────────────────────────────────────────
    def _feature_counts(event_type: str) -> pd.Series:
        return (
            el[el['event_type'].str.upper() == event_type]
            .drop_duplicates(['index_visit_occurrence_id', 'activity'])
            .groupby('activity')['index_visit_occurrence_id'].count()
            .sort_values(ascending=False)
        )

    drug_all  = _feature_counts('DRUG')
    proc_all  = _feature_counts('PROCEDURE')
    cond_all  = _feature_counts('CONDITION')

    drug_sel  = drug_all[drug_all >= min_count].head(TOP_N_DRUGS)
    proc_sel  = proc_all[proc_all >= min_count].head(TOP_N_PROCEDURES)
    cond_sel  = cond_all[cond_all >= min_count]

    w('FEATURES SELECTED BY PIPELINE')
    w('=' * 60)
    w(f'  Frequency threshold : ≥{ACTIVITY_MIN_FREQUENCY*100:.0f}% of visits  (= {min_count:,} visits)')
    w('')
    w(f'  Drugs      — unique in data: {len(drug_all):,}  |  passing threshold: {(drug_all >= min_count).sum()}  |  selected (top {TOP_N_DRUGS}): {len(drug_sel)}')
    w(f'  Procedures — unique in data: {len(proc_all):,}  |  passing threshold: {(proc_all >= min_count).sum()}  |  selected (top {TOP_N_PROCEDURES}): {len(proc_sel)}')
    w(f'  Conditions — unique in data: {len(cond_all):,}  |  passing threshold: {len(cond_sel)}  (no cap)')
    w('')

    def _feature_block(label: str, selected: pd.Series) -> None:
        w(f'  {label}:')
        for rank, (name, cnt) in enumerate(selected.items(), 1):
            pct = cnt / n_visits * 100
            w(f'    {rank:>2}. {name}  ({cnt:,} visits, {pct:.1f}%)')
        w('')

    _feature_block(f'Drugs selected (top {TOP_N_DRUGS})', drug_sel)
    _feature_block(f'Procedures selected (top {TOP_N_PROCEDURES})', proc_sel)
    _feature_block(f'Conditions selected', cond_sel)

    # ── Write file ────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'eda_summary.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'   EDA saved to {out_path}')
