from copy import deepcopy

from ipfn.ipfn import ipfn
from pandas import DataFrame, merge
from scipy.optimize import minimize


def pop_pivot(
    base_pop: DataFrame,
    pivot_cfg: dict = {"index": ["sex", "ethnicity"], "columns": ["age"]},
    values_name: str = "count",
):
    """
    Pivots the population data based on the provided configuration.

    Args:
        base_pop (DataFrame): The base population DataFrame.
        pivot_cfg (dict, optional): Configuration for pivoting.
            Defaults to {"index": ["sex", "ethnicity"], "columns": ["age"]}.
        values_name (str, optional): Name of the values column. Defaults to "count".

    Returns:
        DataFrame: The pivoted population DataFrame.
    """
    base_pop_pivot = base_pop.pivot_table(
        index=pivot_cfg["index"],
        columns=pivot_cfg["columns"],
        values=values_name,
        aggfunc="sum",
        fill_value=0,
    )

    col_names = [None] * len(pivot_cfg["columns"])
    ind_names = [None] * len(pivot_cfg["index"])

    if len(col_names) > 1:
        base_pop_pivot.columns.names = col_names
    else:
        base_pop_pivot.columns.name = col_names[0]

    if len(ind_names) > 1:
        base_pop_pivot.index.names = ind_names
    else:
        base_pop_pivot.index.name = ind_names[0]

    return base_pop_pivot.astype(float)


def ipf_adjustment(base_pop_input: DataFrame, constrains: dict):
    """
    Adjusts the population data using the Iterative Proportional Fitting (IPF) algorithm.

    Args:
        base_pop_input (DataFrame): The input population DataFrame.
        constrains (dict): Dictionary of constraints for IPF adjustment.

    Returns:
        DataFrame: The adjusted population DataFrame.
    """
    base_pop = deepcopy(base_pop_input)

    constrain_indices = []
    constrain_values = []
    for proc_constrain_key in constrains:

        if constrains[proc_constrain_key] is None:
            continue

        if proc_constrain_key == "index":
            proc_index = 0
        elif proc_constrain_key == "columns":
            proc_index = 1

        constrain_indices.append([proc_index])
        constrain_values.append(constrains[proc_constrain_key])

    IPF = ipfn(
        base_pop.values,
        constrain_values,
        constrain_indices,
        convergence_rate=1e-6,
    )
    synthetic_population_matrix = IPF.iteration()

    base_pop = base_pop.astype(float)

    base_pop.values[:] = synthetic_population_matrix

    return base_pop


def postproc(
    pop_orig: DataFrame,
    pop_updated: DataFrame,
    colnames: list = ["sex", "ethnicity", "age"],
    apply_scaler: bool = True,
):
    """
    Post-processes the IPF adjusted population data.

    Args:
        pop_orig (DataFrame): The original population DataFrame.
        pop_updated (DataFrame): The IPF adjusted population DataFrame.
        colnames (list, optional): List of column names to merge on.
            Defaults to ["sex", "ethnicity", "age"].
        apply_scaler (bool, optional): Whether to apply scaling. Defaults to True.

    Returns:
        DataFrame: The post-processed population DataFrame.
    """

    def _objective(df: DataFrame, scaling_factor: float):
        return abs(df["count"].sum() - (df["count_adjusted"] * scaling_factor).sum())

    pop_updated = pop_updated.stack().reset_index()
    pop_updated.columns = colnames + ["count"]
    pop_updated = pop_updated[pop_updated["count"] != 0]

    # Reset the index of the final DataFrame
    pop_updated = pop_updated.reset_index(drop=True)

    merged_data = merge(
        pop_orig,
        pop_updated,
        on=colnames,
        suffixes=("", "_adjusted"),
        how="outer",
    )

    if apply_scaler:
        result = minimize(lambda x: _objective(merged_data, x), x0=1.0)
        optimal_scaling_factor = result.x[0]
        merged_data["count_adjusted"] *= optimal_scaling_factor

    merged_data = merged_data[colnames + ["count_adjusted"]]

    merged_data = merged_data.rename(columns={"count_adjusted": "count"})

    return merged_data
