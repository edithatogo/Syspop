from logging import getLogger
from uuid import uuid4

from pandas import DataFrame, Series

from syspop.python.utils import select_place_with_contstrain

logger = getLogger()


def create_households(
    household_data: DataFrame,
    address_data: DataFrame,
    areas: list,
) -> DataFrame:
    """
    Disaggregates household composition data to create individual household records,
    each assigned a unique ID and specific geographic coordinates.

    The input `household_data` provides counts of households with specific
    numbers of adults and children within an area. This function expands these
    counts into individual household entries. For each created household, it
    samples a random address (latitude, longitude) from `address_data`
    (filtered for the household's area).

    Args:
        household_data (DataFrame): Aggregated household data. Expected columns:
            'area', 'adults', 'children', 'value' (count of such households).
            Can optionally include an 'ethnicity' column.
        address_data (DataFrame): A DataFrame of available addresses with 'area',
                                  'latitude', and 'longitude' columns.
        areas (list): A list of area IDs to process. Both `household_data` and
                      `address_data` will be filtered by these areas.

    Returns:
        DataFrame: A DataFrame where each row represents an individual household.
                   Columns include 'area', 'adults', 'children', 'latitude',
                   'longitude', 'household' (a unique ID), and 'ethnicity'
                   (if present in input).
    """
    households = []

    household_data = household_data[household_data["area"].isin(areas)]
    address_data = address_data[address_data["area"].isin(areas)]

    # Loop through each row in the original DataFrame
    for _, row in household_data.iterrows():
        area = row["area"]
        adults = row["adults"]
        children = row["children"]
        count = row["value"]

        ethnicity = None
        if "ethnicity" in row:
            ethnicity = row["ethnicity"]

        proc_address_data_area = address_data[address_data["area"] == area]

        # Create individual records for each household
        for _ in range(count):
            proc_address_data = proc_address_data_area.sample(n=1)

            proc_hhd_info = {
                "area": int(area),
                "adults": int(adults),
                "children": int(children),
                "latitude": float(proc_address_data.latitude),
                "longitude": float(proc_address_data.longitude),
                "household": str(uuid4())[:6],  # Create a 6-digit unique ID
            }

            if ethnicity is not None:
                proc_hhd_info["ethnicity"] = ethnicity

            households.append(proc_hhd_info)

    return DataFrame(households)


def place_agent_to_household(households: DataFrame, agent: Series) -> tuple:
    """
    Assigns an agent to a suitable household, updating the household's capacity.

    The function determines if the agent is an adult (>=18) or child.
    It then tries to find a household in the agent's 'area' that has space
    for that type of agent.
    If 'ethnicity' is a column in `households`, it attempts to place the
    agent in a household matching their ethnicity with higher probability.
    If no suitable household is found in the agent's area, a random household
    is chosen from the entire `households` DataFrame.

    Once a household is selected, its capacity for the agent type is decremented.
    The agent is updated with the 'household' ID.

    Args:
        households (DataFrame): DataFrame of available households. Expected columns:
            'area', 'adults' (capacity), 'children' (capacity), 'household' (ID).
            Can optionally include 'ethnicity'.
        agent (Series): The agent to be placed. Expected attributes: 'age', 'area',
                        and 'ethnicity' (if ethnicity matching is used).

    Returns:
        tuple:
            - agent (Series): The updated agent Series with 'household' attribute set.
            - households (DataFrame): The updated households DataFrame with decremented capacity.
    """
    agent_type = "adults" if agent.age >= 18 else "children"

    selected_households = households[
        (households[agent_type] >= 1) & (households["area"] == agent.area)
    ]
    if len(selected_households) > 0:

        if "ethnicity" in selected_households:
            selected_household = select_place_with_contstrain(
                selected_households,
                "ethnicity",
                agent.ethnicity,
                list(selected_households.ethnicity.unique()),
            )
        else:
            selected_household = selected_households.sample(n=1)

        households.at[selected_household.index[0], agent_type] -= 1
    else:
        selected_household = households.sample(n=1)

    agent["household"] = selected_household.household.values[0]
    return agent, households
