from os.path import exists, join
from pickle import load as pickle_load

from numpy.random import choice
from pandas import DataFrame, read_parquet

from syspop.python import NZ_DATA_DEFAULT


def new_zealand(
    data_dir: str = NZ_DATA_DEFAULT, apply_pseudo_ethnicity: bool = False
) -> dict:
    """
    Loads various New Zealand specific datasets required for synthetic population generation.

    This function reads multiple Parquet files from the specified `data_dir`.
    These files include data on population structure, geography, household
    composition, commute patterns, work, education, and shared spaces.
    Optionally, it can add a pseudo 'ethnicity' column to the household
    composition data for testing purposes.

    Args:
        data_dir (str, optional): The directory path where the Parquet input files
            are located. Defaults to `NZ_DATA_DEFAULT`.
        apply_pseudo_ethnicity (bool, optional): If True, a pseudo 'ethnicity'
            column is added to the 'household_composition' DataFrame using
            `add_pseudo_hhd_ethnicity`. Defaults to False.

    Returns:
        dict: A dictionary where keys are dataset names (e.g., "population_structure",
              "geography_hierarchy") and values are the loaded pandas DataFrames.
    """
    nz_data = {}

    nz_data["geography"] = {}
    for item in [
        "population_structure",
        "geography_hierarchy",
        "geography_location",
        "geography_address",
        "household_composition",
        "commute_travel_to_work",
        "commute_travel_to_school",
        "work_employee",
        "work_employer",
        "work_income",
        "school",
        "kindergarten",
        "hospital",
        "shared_space_bakery",
        "shared_space_cafe",
        "shared_space_department_store",
        "shared_space_fast_food",
        "shared_space_park",
        "shared_space_pub",
        "shared_space_restaurant",
        "shared_space_supermarket",
        "shared_space_wholesale",
    ]:
        proc_path = join(data_dir, f"{item}.parquet")
        if exists(proc_path):
            nz_data[item] = read_parquet(proc_path)

    if apply_pseudo_ethnicity:
        nz_data["household_composition"] = add_pseudo_hhd_ethnicity(
            nz_data["household_composition"]
        )

    return nz_data


def add_pseudo_hhd_ethnicity(
    household_composition_data: DataFrame,
    ethnicities: list = ["European", "Maori", "Pacific", "Asian", "MELAA"],
    weights: list = [0.6, 0.15, 0.1, 0.12, 0.03],
) -> DataFrame:
    """
    Adds a pseudo 'ethnicity' column to a household composition DataFrame.

    The ethnicity is assigned randomly based on the provided `ethnicities` list
    and their corresponding `weights`. This is primarily used for testing or
    when actual ethnicity data for households is not available.

    Args:
        household_composition_data (DataFrame): The input DataFrame to which
            the 'ethnicity' column will be added.
        ethnicities (list, optional): A list of ethnicity strings to choose from.
            Defaults to ["European", "Maori", "Pacific", "Asian", "MELAA"].
        weights (list, optional): A list of weights corresponding to `ethnicities`.
            Must sum to 1.0. Defaults to [0.6, 0.15, 0.1, 0.12, 0.03].

    Returns:
        DataFrame: The input DataFrame with an added 'ethnicity' column.
    """
    household_composition_data["ethnicity"] = choice(
        ethnicities, size=len(household_composition_data), p=weights
    )
    return household_composition_data


def load_llm_diary(data_dir: str = NZ_DATA_DEFAULT) -> dict:
    """
    Loads pre-generated LLM diary data from a pickle file.

    The pickle file is expected to be named "llm_diary.pickle" and located
    in the specified `data_dir`.

    Args:
        data_dir (str, optional): The directory where "llm_diary.pickle" is located.
            Defaults to `NZ_DATA_DEFAULT`.

    Returns:
        dict: The loaded LLM diary data. The structure of this dictionary
              depends on how it was originally created and saved.
    """
    with open(f"{data_dir}/llm_diary.pickle", "rb") as fid:
        llm_diary_data = pickle_load(fid)
    return llm_diary_data
