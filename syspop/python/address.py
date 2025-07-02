from datetime import datetime
from logging import getLogger

from pandas import DataFrame

logger = getLogger()



def assign_place_to_address(
    address_type: str,
    pop_data_input: DataFrame,
    address_data_input: DataFrame,
    proc_area: int,
) -> list[str]:
    """
    Assigns specific addresses to unique place names (e.g., households, companies)
    within a processed area.

    For each unique `address_type` (e.g., unique household ID, unique company name)
    found in `pop_data_input`, this function samples a random address
    (latitude, longitude) from `address_data_input` (which should be pre-filtered
    for `proc_area`). It then creates a string representation combining the
    original place name, its new coordinates, and the area ID.

    Args:
        address_type (str): The column name in `pop_data_input` that holds the
                            unique names/IDs of places to be assigned addresses
                            (e.g., "household", "company").
        pop_data_input (DataFrame): A DataFrame (filtered for `proc_area`)
                                    containing the unique place names/IDs under the
                                    column specified by `address_type`.
        address_data_input (DataFrame): A DataFrame of available addresses
                                        (filtered for `proc_area`) with 'latitude'
                                        and 'longitude' columns.
        proc_area (int): The specific area ID being processed.

    Returns:
        list[str]: A list of strings, where each string is formatted as
                   "place_name,latitude,longitude,area_id".
                   Returns an empty list if `address_data_input` is empty.
    """

    all_address = []
    all_address_names = list(pop_data_input[address_type].unique())

    for proc_address_name in all_address_names:
        if len(address_data_input) > 0:
            proc_address = address_data_input.sample(n=1)
            all_address.append(
                f"{proc_address_name}, {round(proc_address['latitude'].values[0], 5)},{round(proc_address['longitude'].values[0], 5)}, {proc_area}"
            )

    return all_address


def add_random_address(
    base_pop: DataFrame,
    address_data: DataFrame,
    address_type: str
) -> DataFrame:
    """
    Assigns random addresses (latitude, longitude) to unique entities (e.g.,
    households, companies) within each area present in the `base_pop` DataFrame.

    This function iterates through each unique area in `base_pop`. For each area,
    it identifies unique entities of `address_type` (e.g., household IDs, company names)
    and assigns them a randomly selected address from the `address_data` (filtered
    for that area). The results are compiled into a new DataFrame.

    Args:
        base_pop (DataFrame): A DataFrame containing the base population. It must
                              have an 'area' column and a column corresponding to
                              `address_type` (e.g., 'household', 'company'). If
                              `address_type` is "company", it's expected to have
                              an 'area_work' column instead of 'area' for filtering.
        address_data (DataFrame): A DataFrame containing available addresses with
                                  'area', 'latitude', and 'longitude' columns.
        address_type (str): The type of entity to assign addresses to. This string
                            is used as a column name in `base_pop` and to name
                            the 'type' column in the output DataFrame.

    Returns:
        DataFrame: A DataFrame with columns ['name', 'latitude', 'longitude', 'area', 'type'],
                   where 'name' is the unique ID/name of the entity, and 'type' is
                   the `address_type`.
    """
    start_time = datetime.utcnow()

    all_areas = list(base_pop["area"].unique())

    results = []

    for i, proc_area in enumerate(all_areas):
        logger.info(f"{i}/{len(all_areas)}: Processing {proc_area}")

        proc_address_data = address_data[address_data["area"] == proc_area]

        area_type = "area"
        if address_type == "company":
            area_type = "area_work"

        proc_pop_data = base_pop[base_pop[area_type] == proc_area]

        processed_address = assign_place_to_address(
            address_type, proc_pop_data, proc_address_data, proc_area
        )

        results.append(processed_address)

    flattened_results = [item for sublist in results for item in sublist]
    results_dict = {"name": [], "latitude": [], "longitude": [], "area": []}
    for proc_result in flattened_results:
        proc_value = proc_result.split(",")
        results_dict["name"].append(proc_value[0])
        results_dict["latitude"].append(float(proc_value[1]))
        results_dict["longitude"].append(float(proc_value[2]))
        results_dict["area"].append(int(proc_value[3]))

    results_df = DataFrame.from_dict(results_dict)
    results_df["type"] = address_type

    results_df["area"] = results_df["area"].astype(int)

    end_time = datetime.utcnow()

    total_mins = round((end_time - start_time).total_seconds() / 60.0, 3)
    logger.info(f"Processing time (address): {total_mins}")

    return results_df
