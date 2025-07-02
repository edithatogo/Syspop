from datetime import datetime
from logging import getLogger

from pandas import DataFrame

logger = getLogger()

def base_pop_wrapper(
    structure_data: DataFrame,
    output_area_filter: list | None,
) -> DataFrame:
    """
    Creates a base population DataFrame by disaggregating summarized structure data.

    The input `structure_data` is expected to have columns like 'area', 'age',
    'gender', 'ethnicity', and a 'value' column indicating the count of
    individuals with that combination of characteristics. This function repeats
    each row `value` times to create a DataFrame where each row represents
    an individual.

    Args:
        structure_data (DataFrame): A DataFrame containing aggregated population
                                    counts for different demographic groups and areas.
                                    Must include a 'value' column for counts and
                                    an 'area' column.
        output_area_filter (list | None): An optional list of area IDs. If provided,
                                          the `structure_data` will be filtered to
                                          include only these areas before processing.

    Returns:
        DataFrame: A disaggregated population DataFrame with columns for each
                   demographic characteristic (e.g., 'area', 'age', 'gender',
                   'ethnicity'), where each row represents one individual.
                   The 'value' column is dropped.
    """

    start_time = datetime.utcnow()
    if output_area_filter is not None:
        structure_data = structure_data[structure_data["area"].isin(output_area_filter)]

    if structure_data is not None:
        df_repeated = structure_data.loc[structure_data.index.repeat(structure_data["value"].astype(int))]
        population = df_repeated.reset_index(drop=True).drop(columns=["value"])

    end_time = datetime.utcnow()
    total_mins = round((end_time - start_time).total_seconds() / 60.0, 2)

    logger.info(f"Processing time (base population): {total_mins} minutes")

    return population
