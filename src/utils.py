import pandas as pd


def get_descendant_ids(concept_ancestor_df: pd.DataFrame,
                       ancestor_ids: list) -> set:

    mask = concept_ancestor_df['ancestor_concept_id'].isin(ancestor_ids)
    descendants = set(
        concept_ancestor_df.loc[mask, 'descendant_concept_id'].unique()
    )
    descendants.update(ancestor_ids)
    return descendants
