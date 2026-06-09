import os
import pandas as pd


DATE_COLS = {
    'visit_occurrence': ['visit_start_date', 'visit_end_date'],
    'condition_occurrence': ['condition_start_date'],
    'drug_exposure': ['drug_exposure_start_date'],
    'procedure_occurrence': ['procedure_date'],
}

USECOLS = {
    'concept_ancestor': ['ancestor_concept_id', 'descendant_concept_id'],
}


def load_omop_table(name: str, data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, f'{name}.csv')
    date_cols = DATE_COLS.get(name, [])
    usecols = USECOLS.get(name, None)
    return pd.read_csv(path, parse_dates=date_cols, usecols=usecols, low_memory=False)


def load_all_tables(data_dir: str) -> dict:
    tables = {}
    for name in ['person', 'visit_occurrence', 'condition_occurrence',
                 'drug_exposure', 'procedure_occurrence', 'concept', 'concept_ancestor']:
        tables[name] = load_omop_table(name, data_dir)
        print(f'  {name}: {len(tables[name]):,} rows')
    return tables
