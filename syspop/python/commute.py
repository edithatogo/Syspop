from logging import getLogger

from numpy.random import choice as numpy_choice
from pandas import DataFrame, Series

logger = getLogger()


def create_commute_probability(
        commute_dataset: DataFrame,
        areas: list,
        commute_type: str = "work") -> DataFrame:
    """
    Calculates commute probabilities for various travel methods between home areas
    and destination areas (work or school).

    The function first filters the `commute_dataset` for the specified `areas`.
    It then calculates the total number of commuters from each 'area_home'.
    Finally, it computes the probability of using each travel method for commuting
    from an 'area_home' to an 'area_{commute_type}' by dividing the sum of people
    using each method by the total commuters from that 'area_home'.

    Args:
        commute_dataset (DataFrame): A DataFrame containing commute counts.
            Expected columns: 'area_home', 'area_{commute_type}' (e.g., 'area_work'),
            and columns for each travel method with their respective counts.
        areas (list): A list of 'area_home' IDs to include in the calculation.
        commute_type (str, optional): The type of commute, used to determine the
            destination area column name (e.g., "work" -> "area_work").
            Defaults to "work".

    Returns:
        DataFrame: A DataFrame with columns 'area_home', 'area_{commute_type}',
                   and columns for each travel method containing the calculated
                   probabilities.
    """
    commute_dataset = commute_dataset[commute_dataset["area_home"].isin(areas)]
    travel_methods = [col for col in commute_dataset.columns if col not in [
            "area_home", f"area_{commute_type}"]]

    total_people = commute_dataset.groupby("area_home")[travel_methods].sum().sum(axis=1)
    area_sums = commute_dataset.groupby(["area_home", f"area_{commute_type}"])[travel_methods].sum()
    return area_sums.div(total_people, axis=0).reset_index()


def assign_agent_to_commute(
        commute_dataset: DataFrame,
        agent: Series,
        commute_type: str = "work",
        include_filters: dict = {}) -> Series:
    """
    Assigns a commute destination area and travel method to an agent.

    The function first checks if the agent meets criteria specified in `include_filters`
    (e.g., age range for work/school). If not, the commute attributes are set to None.
    Otherwise, it filters the `commute_dataset` for the agent's home area,
    calculates the total probability for each destination, selects a destination
    area based on these probabilities, and then selects a travel method to that
    destination based on the method probabilities.

    Args:
        commute_dataset (DataFrame): A DataFrame of commute probabilities,
            as generated by `create_commute_probability`. Expected columns:
            'area_home', 'area_{commute_type}', and columns for travel methods.
        agent (Series): A pandas Series representing the agent. Must have an 'area'
                        attribute (home area) and attributes corresponding to keys
                        in `include_filters`.
        commute_type (str, optional): The type of commute ("work" or "school").
            This determines the names of the assigned attributes (e.g., "area_work",
            "travel_method_work"). Defaults to "work".
        include_filters (dict, optional): A dictionary where keys are agent
            attributes (e.g., "age") and values are lists of tuples, each
            defining an inclusive range [(min1, max1), (min2, max2)]. An agent
            is considered for commute assignment only if their attribute value
            falls within ANY of these ranges for ALL filter keys.
            Defaults to {}.

    Returns:
        Series: The input `agent` Series, updated with 'area_{commute_type}'
                (int or None) and 'travel_method_{commute_type}' (str or None).
    """
    for include_key in include_filters:
        proc_filters = include_filters[include_key]
        for proc_filter in proc_filters:
            if agent[include_key] < proc_filter[0] or agent[include_key] > proc_filter[1]:
                agent[f"area_{commute_type}"] = None
                agent[f"travel_method_{commute_type}"] = None
                return agent

    proc_commute_dataset = commute_dataset[commute_dataset.area_home == agent.area]
    proc_commute_dataset["total"] = proc_commute_dataset.drop(
        columns=["area_home", f"area_{commute_type}"]).sum(axis=1)
    selected_row = proc_commute_dataset.loc[
        numpy_choice(proc_commute_dataset.index, p=proc_commute_dataset["total"])]
    selected_area = int(selected_row[f"area_{commute_type}"])
    selected_row = selected_row.drop(["area_home", f"area_{commute_type}", "total"])
    travel_method = numpy_choice(
        selected_row.index, p=selected_row /selected_row.sum())

    agent[f"area_{commute_type}"] = selected_area
    agent[f"travel_method_{commute_type}"] = travel_method

    return agent
