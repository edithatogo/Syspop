from pandas import DataFrame, read_csv

from funcs import POPULATION_CODE


def read_population_structure(population_structure_data_path: str) -> DataFrame:
    """
    Reads and processes a population structure CSV file.

    The function reads the specified CSV, selects relevant columns ('sa2',
    'ethnicity', 'age', 'gender', 'value'), renames 'sa2' to 'area',
    replaces coded values for 'ethnicity' and 'gender' with their
    string representations based on POPULATION_CODE, and resets the index.

    Args:
        population_structure_data_path (str): The file path to the population
                                              structure CSV data.

    Returns:
        DataFrame: A DataFrame with columns ['area', 'ethnicity', 'age',
                   'gender', 'value'].
    """
    df = read_csv(population_structure_data_path)[["sa2", "ethnicity", "age", "gender", "value"]]
    df = df.rename(columns={"sa2": "area"})
    df = df.replace(POPULATION_CODE).reset_index()
    df = df.drop(columns=["index"])
    return df
