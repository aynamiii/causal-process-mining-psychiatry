import pandas as pd
from config import TOP_N_DRUGS, THERAPY_SNOMED_CONCEPTS
from src.utils import get_descendant_ids


# Charlson Comorbidity Index
CHARLSON = {
    'MI':             (1, ['I21', 'I22', 'I25.2']),
    'CHF':            (1, ['I09.9', 'I11.0', 'I13.0', 'I13.2', 'I25.5',
                            'I42', 'I43', 'I50', 'P29.0']),
    'PVD':            (1, ['I70', 'I71', 'I73', 'I77.1', 'I79',
                            'K55.1', 'K55.8', 'K55.9', 'Z95.8', 'Z95.9']),
    'Stroke':         (1, ['G45', 'G46', 'H34.0', 'I60', 'I61', 'I62',
                            'I63', 'I64', 'I65', 'I66', 'I67', 'I68']),
    'Dementia':       (1, ['F00', 'F01', 'F02', 'F03', 'F05.1', 'G30', 'G31.1']),
    'COPD':           (1, ['I27.8', 'I27.9', 'J40', 'J41', 'J42', 'J43', 'J44',
                            'J45', 'J46', 'J47', 'J60', 'J61', 'J62', 'J63',
                            'J64', 'J65', 'J66', 'J67']),
    'Rheumatic':      (1, ['M05', 'M06', 'M31.5', 'M32', 'M33', 'M34',
                            'M35.1', 'M35.3', 'M36.0']),
    'PUD':            (1, ['K25', 'K26', 'K27', 'K28']),
    'Liver_mild':     (1, ['B18', 'K70.0', 'K70.1', 'K70.2', 'K70.3', 'K70.9',
                            'K71', 'K73', 'K74', 'K76.0', 'K76.2', 'K76.3',
                            'K76.4', 'K76.8', 'K76.9', 'Z94.4']),
    'Diabetes':       (1, ['E10.0', 'E10.1', 'E10.6', 'E10.8', 'E10.9',
                            'E11.0', 'E11.1', 'E11.6', 'E11.8', 'E11.9',
                            'E12', 'E13', 'E14']),
    'Diabetes_compl': (2, ['E10.2', 'E10.3', 'E10.4', 'E10.5', 'E10.7',
                            'E11.2', 'E11.3', 'E11.4', 'E11.5', 'E11.7']),
    'Hemiplegia':     (2, ['G04.1', 'G11.4', 'G80.1', 'G80.2', 'G81', 'G82',
                            'G83.0', 'G83.1', 'G83.2', 'G83.3', 'G83.9']),
    'Renal':          (2, ['I12.0', 'I13.1', 'N03.2', 'N03.3', 'N03.4',
                            'N03.5', 'N03.6', 'N03.7', 'N05.2', 'N05.3',
                            'N05.4', 'N05.5', 'N05.6', 'N05.7', 'N18', 'N19',
                            'N25.0', 'Z49.0', 'Z49.1', 'Z49.2', 'Z94.0', 'Z99.2']),
    'Cancer':         (2, ['C0', 'C1', 'C2', 'C3', 'C4', 'C5',
                            'C6', 'C7', 'C8', 'C9']),
    'Liver_severe':   (3, ['I85.0', 'I85.9', 'I86.4', 'I98.2', 'K70.4',
                            'K71.1', 'K72.1', 'K72.9', 'K76.5', 'K76.6', 'K76.7']),
    'Metastatic':     (6, ['C77', 'C78', 'C79', 'C80']),
    'HIV':            (6, ['B20', 'B21', 'B22', 'B24']),
}


def compute_cci(condition_df: pd.DataFrame) -> pd.Series:

    cond = condition_df[['person_id', 'condition_source_value']].dropna().copy()
    cond['icd'] = cond['condition_source_value'].astype(str).str.strip().str.upper()
    scores = {}
    for _, (weight, prefixes) in CHARLSON.items():
        pattern = '|'.join(f'^{p}' for p in prefixes)
        for pid in cond.loc[cond['icd'].str.match(pattern, na=False), 'person_id'].unique():
            scores[pid] = scores.get(pid, 0) + weight
    return pd.Series(scores, name='cci_score')


def _build_concept_lookup(concept_df: pd.DataFrame) -> dict:

    lookup = {}
    for _, row in concept_df.iterrows():
        name = str(row['concept_name']).strip().lower()
        name = name.replace(' ', '_').replace('/', '_').replace('-', '_')
        name = ''.join(c for c in name if c.isalnum() or c == '_')
        name = name[:40].rstrip('_')
        lookup[int(row['concept_id'])] = name
    return lookup


def _resolve_therapy_concepts(concept_ancestor_df: pd.DataFrame) -> dict:

    resolved = {}
    for therapy_name, ancestor_ids in THERAPY_SNOMED_CONCEPTS.items():
        concept_ids = get_descendant_ids(concept_ancestor_df, ancestor_ids)
        print(f'  {therapy_name}: {len(concept_ids):,} concept(s) found')
        resolved[therapy_name] = concept_ids
    return resolved


def build_features(
    visit_df: pd.DataFrame,
    person_df: pd.DataFrame,
    condition_df: pd.DataFrame,
    drug_df: pd.DataFrame,
    procedure_df: pd.DataFrame,
    concept_df: pd.DataFrame,
    concept_ancestor_df: pd.DataFrame,
    top_n_drugs: int = TOP_N_DRUGS,
) -> tuple[pd.DataFrame, list, list]:

    # Build concept lookup dict (concept_id -> readable name)
    concept_lookup = _build_concept_lookup(concept_df)

    # Resolve therapy concept IDs via concept_ancestor expansion
    therapy_map = _resolve_therapy_concepts(concept_ancestor_df)

    # Base
    df = visit_df[['visit_occurrence_id', 'person_id',
                   'visit_start_date', 'visit_end_date', 'visit_concept_id']].copy()

    # Demographics
    df = df.merge(
        person_df[['person_id', 'year_of_birth', 'gender_concept_id']],
        on='person_id', how='left'
    )
    df['visit_start_date'] = pd.to_datetime(df['visit_start_date'])
    df['visit_end_date'] = pd.to_datetime(df['visit_end_date'])
    df['age'] = df['visit_start_date'].dt.year - df['year_of_birth']
    df['los_days'] = (df['visit_end_date'] - df['visit_start_date']).dt.days.clip(lower=0)
    df['gender'] = df['gender_concept_id'].map({8507: 'M', 8532: 'F'}).fillna('U')

    # Visit type: resolved from concept table, with a clear label for unknowns
    df['visit_type'] = df['visit_concept_id'].map(concept_lookup).fillna('other')

    df['age_group'] = pd.cut(
        df['age'], bins=[0, 40, 55, 65, 75, 150],
        labels=['<40', '40-54', '55-64', '65-74', '75+']
    )

    # CCI
    df['cci_score'] = df['person_id'].map(compute_cci(condition_df)).fillna(0).astype(int)

    # Top-N drug binary indicators (exclude concept_id = 0)
    drug_valid = drug_df[
        (drug_df['visit_occurrence_id'].notna()) &
        (drug_df['drug_concept_id'] != 0)
    ].copy()
    top_drugs = drug_valid['drug_concept_id'].value_counts().head(top_n_drugs).index
    drug_filt = drug_valid[drug_valid['drug_concept_id'].isin(top_drugs)].copy()

    # Use concept name if available, otherwise fall back to concept ID
    drug_filt['col'] = drug_filt['drug_concept_id'].apply(
        lambda cid: 'drug_' + concept_lookup.get(int(cid), str(int(cid)))
    )
    drug_dum = (drug_filt.groupby(['visit_occurrence_id', 'col'])
                 .size().unstack(fill_value=0).clip(upper=1))
    drug_dum.columns.name = None
    drug_cols = list(drug_dum.columns)
    df = df.merge(drug_dum, on='visit_occurrence_id', how='left')
    df[drug_cols] = df[drug_cols].fillna(0).astype(int)

    # Therapy binary indicators (using dynamically resolved concept IDs)
    therapy_cols = []
    for therapy_name, concept_ids in therapy_map.items():
        visits_with = set(
            procedure_df.loc[
                procedure_df['procedure_concept_id'].isin(concept_ids) &
                procedure_df['visit_occurrence_id'].notna(),
                'visit_occurrence_id'
            ].unique()
        )
        df[therapy_name] = df['visit_occurrence_id'].isin(visits_with).astype(int)
        therapy_cols.append(therapy_name)

    return df.reset_index(drop=True), drug_cols, therapy_cols
