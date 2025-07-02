from copy import deepcopy

from pandas import DataFrame
from pulp import lpSum


def add_constrains_shared_space(
    costfunc,
    constrain_var: str,
    population_data_input: DataFrame,
    constrain_data_input: DataFrame,
    population_constrain_map_keys: dict = {"age": ["adult", "child"]},
) -> tuple:
    """
    Adds constraints to a PuLP cost function based on shared space data.

    This function is designed to link population data (e.g., by age) to
    shared space usage data (e.g., number of adults and children in a space).
    It ensures that the sum of individuals from the population data matching
    certain criteria (e.g., 'adult' or 'child' age groups) equals the
    sum of individuals using a shared space with corresponding characteristics.

    Args:
        costfunc (dict): A dictionary containing the PuLP LpProblem object
            (costfunc['costfunc']) and LpVariable dictionaries
            (costfunc['constrain_vars']).
        constrain_var (str): The key in `costfunc['constrain_vars']` that
            holds the LpVariables for the shared space being constrained.
        population_data_input (DataFrame): DataFrame containing population counts,
            expected to have a column specified in `population_constrain_map_keys`
            (e.g., 'age') and a 'count' column.
        constrain_data_input (DataFrame): DataFrame containing shared space usage
            counts, expected to have columns specified in the values of
            `population_constrain_map_keys` (e.g., ['adult', 'child']) and
            a 'count' column.
        population_constrain_map_keys (dict, optional): A dictionary mapping a
            column name in `population_data_input` to a list of column names in
            `constrain_data_input`. Currently supports only one key-value pair.
            Defaults to {"age": ["adult", "child"]}.

    Returns:
        tuple: The modified `costfunc` dictionary with added constraints.

    Raises:
        Exception: If `population_constrain_map_keys` has more than one key.
    """

    if len(population_constrain_map_keys.keys()) > 1:
        raise Exception("Population and constrain mapping only support one key")

    population_constrain_map_key = list(population_constrain_map_keys.keys())[0]
    population_constrain_map_values = list(population_constrain_map_keys.values())[0]

    population_data = deepcopy(
        population_data_input[[population_constrain_map_key, "count"]]
    ).reset_index()

    population_data = (
        population_data.groupby(population_constrain_map_key)
        .agg({"index": lambda x: list(x), "count": "sum"})
        .reset_index()
    )

    constrain_data = deepcopy(
        constrain_data_input[population_constrain_map_values + ["count"]]
    ).reset_index()

    constrain_data = (
        constrain_data.groupby(population_constrain_map_values)
        .agg({"index": lambda x: list(x), "count": "sum"})
        .reset_index()
    )

    for proc_key in population_constrain_map_values:

        proc_constrain_terms = []
        for _, proc_constrain_data in constrain_data[
            [proc_key] + ["index", "count"]
        ].iterrows():
            for proc_index in proc_constrain_data["index"]:
                proc_constrain_terms.append(
                    costfunc["constrain_vars"][constrain_var][proc_index]
                    * proc_constrain_data[proc_key]
                )

        proc_population_terms = [
            costfunc["constrain_vars"]["population_data"][i]
            for i in population_data[
                population_data[population_constrain_map_key] == proc_key
            ]["index"].iloc[0]
        ]

        costfunc["costfunc"] += lpSum(proc_constrain_terms) == lpSum(
            proc_population_terms
        )

    return costfunc
