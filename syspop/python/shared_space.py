from logging import getLogger
from uuid import uuid4

from numpy import arange as numpy_arrange
from numpy import argsort as numpy_argsort
from numpy.random import choice as numpy_choice
from pandas import DataFrame, Series
from scipy.spatial.distance import cdist

from syspop.python import SHARED_SPACE_NEAREST_DISTANCE_KM

logger = getLogger()


def create_shared_data(shared_space_data: DataFrame, proc_shared_space_name: str) -> DataFrame:
    """
    Transforms raw shared space data into a structured DataFrame with unique IDs.

    Takes a DataFrame with 'area', 'latitude', and 'longitude' for shared spaces,
    and for each row, creates a new record with a unique ID. The unique ID is
    stored in a column named after `proc_shared_space_name`.

    Args:
        shared_space_data (DataFrame): Input DataFrame with at least 'area',
                                       'latitude', and 'longitude' columns.
        proc_shared_space_name (str): The name to use for the new column that
                                      will store the unique ID for each shared space
                                      (e.g., "supermarket_id", "restaurant_id").

    Returns:
        DataFrame: A new DataFrame where each row represents a shared space.
                   Columns: 'area' (int), `proc_shared_space_name` (str, unique ID),
                   'latitude' (float), 'longitude' (float).
    """
    shared_space_data = shared_space_data[["area", "latitude", "longitude"]]

    shared_space_datas = []
    for _, row in shared_space_data.iterrows():
        shared_space_datas.append({
            "area": int(row.area),
            proc_shared_space_name: str(uuid4())[:6],
            "latitude": float(row.latitude),
            "longitude": float(row.longitude),
        })

    return DataFrame(shared_space_datas)


def place_agent_to_shared_space_based_on_area(
        shared_space_data: DataFrame,
        agent: Series,
        shared_space_type: str,
        filter_keys: list = [],
        name_key: str = "id",
        weight_key: str | None = None,
        shared_space_type_convert: dict | None = None) -> Series:
    """
    Assigns an agent to a specific shared space based on their assigned area for that
    shared space type and other optional filters.

    If the agent has an assigned area for the given `shared_space_type` (e.g.,
    `agent['area_supermarket']` is not None), this function filters the
    `shared_space_data` to that area. It then applies further filtering based
    on `filter_keys` (matching agent attributes to shared space attributes).
    A specific shared space is then sampled from the filtered list, optionally
    using `weight_key` for weighted sampling. The ID of the selected space is
    assigned to the agent's attribute corresponding to `shared_space_type`
    (or its converted name via `shared_space_type_convert`).

    Args:
        shared_space_data (DataFrame): DataFrame containing all available shared spaces.
            Must have a column `area_{shared_space_type}` and a column specified by `name_key`.
            Can also have columns matching `filter_keys` and `weight_key`.
        agent (Series): The agent to be assigned. Must have an attribute
            `area_{shared_space_type}` and attributes for any `filter_keys`.
        shared_space_type (str): The generic type of shared space (e.g., "supermarket").
        filter_keys (list, optional): List of attribute names to use for exact
            matching between agent and shared space data. For range filters,
            the shared_space_data should have columns like `key_min` and `key_max`.
            Defaults to [].
        name_key (str, optional): The column name in `shared_space_data` that holds
            the unique ID of the shared space. Defaults to "id".
        weight_key (str | None, optional): If provided, the column name in
            `shared_space_data` to use for weighted sampling. Defaults to None.
        shared_space_type_convert (dict | None, optional): A dictionary to map
            `shared_space_type` to a different attribute name on the agent.
            E.g., {"work": "employer"}. Defaults to None.

    Returns:
        Series: The updated agent Series with the specific shared space ID assigned.
                If no suitable space is found, "Unknown" is assigned. If the agent
                had no area assigned for this space type, the attribute remains unchanged
                or is set based on previous assignments.
    """
    selected_space_id = None

    if agent[f"area_{shared_space_type}"] is not None:

        selected_spaces = shared_space_data[
            shared_space_data[f"area_{shared_space_type}"] ==
            agent[f"area_{shared_space_type}"]
        ]

        for proc_filter_key in filter_keys:
            if proc_filter_key in shared_space_data:
                selected_spaces = selected_spaces[
                    agent[proc_filter_key] == selected_spaces[proc_filter_key]
                ]
            else:
                selected_spaces = selected_spaces[
                    (agent[proc_filter_key] >= selected_spaces[f"{proc_filter_key}_min"]) &
                    (agent[proc_filter_key] <= selected_spaces[f"{proc_filter_key}_max"])
                ]
        if len(selected_spaces) == 0:
            selected_space_id = "Unknown"
        else:
            if weight_key is None:
                selected_space_id = selected_spaces.sample(
                    n=1)[name_key].values[0]
            else:
                selected_space_id = selected_spaces.loc[numpy_choice(
                    selected_spaces.index,
                    p = selected_spaces[weight_key] / selected_spaces[weight_key].sum())][name_key]



    if shared_space_type_convert is not None:
        shared_space_type = shared_space_type_convert[shared_space_type]

    agent[shared_space_type] = selected_space_id

    return agent



def find_nearest_shared_space_from_household(
        household_data: DataFrame,
        shared_space_address: DataFrame,
        geography_location: DataFrame,
        shared_space_type: str,
        n: int =2) -> DataFrame:
    """
    Finds the N nearest shared spaces for each unique household area and adds them
    as a comma-separated string to the geography_location DataFrame.

    It calculates Euclidean distances between SA2 area centroids (from `geography_location`
    filtered by unique areas in `household_data`) and all shared space addresses.
    For each household area, it identifies the `n` nearest shared spaces.
    If a shared space is beyond a type-specific maximum distance (from
    `SHARED_SPACE_NEAREST_DISTANCE_KM`), it's excluded.

    Args:
        household_data (DataFrame): DataFrame with household locations, must have 'area'.
        shared_space_address (DataFrame): DataFrame of shared spaces with 'latitude',
            'longitude', and a column named `shared_space_type` containing IDs.
        geography_location (DataFrame): DataFrame of SA2 area centroids with 'area',
            'latitude', 'longitude'.
        shared_space_type (str): The type of shared space (e.g., "supermarket"),
            used as a key in `SHARED_SPACE_NEAREST_DISTANCE_KM` and as the
            column name in `shared_space_address` holding the IDs.
        n (int, optional): The number of nearest shared spaces to find. Defaults to 2.

    Returns:
        DataFrame: A subset of `geography_location` (for areas present in
                   `household_data`) with an added column named `shared_space_type`.
                   This new column contains a comma-separated string of the IDs of
                   the N nearest shared spaces, or "Unknown" if none are found
                   within the distance threshold.
    """
    updated_src_data = geography_location[
        geography_location["area"].isin(household_data.area.unique())]
    # Extract latitude and longitude as numpy arrays
    coords1 = updated_src_data[["latitude", "longitude"]].values
    coords2 = shared_space_address[["latitude", "longitude"]].values

    # Compute distances:
    # distances will be a matrix where each element [i, j] represents the Euclidean distance
    # between the i-th point in coords1 and the j-th point in coords2.
    distances = cdist(
        coords1,
        coords2,
        metric="euclidean")

    # Find the nearest n indices for each row in household_location_data:
    # nearest_indices is an array where each row contains the indices of the n closest points (in coords2)
    # to the corresponding point in coords1.
    nearest_indices = numpy_argsort(distances, axis=1)[:, :n]

    distance_value = distances[
        numpy_arrange(nearest_indices.shape[0])[:, None], nearest_indices
    ]
    nearest_names = []
    total_missing = 0
    totals_expected = coords1.shape[0] * n # number of household area * number of nearest points
    for i, indices in enumerate(nearest_indices):
        proc_names = []
        for j, index in enumerate(indices):
            proc_dis = distance_value[i, j]
            if proc_dis > SHARED_SPACE_NEAREST_DISTANCE_KM[shared_space_type] / 110.0:
                total_missing += 1
                continue
            proc_names.append(shared_space_address.loc[index][shared_space_type])

        if len(proc_names) == 0:
            nearest_names.append("Unknown")
        else:
            nearest_names.append(", ".join(proc_names))

    logger.info(f"* Missing {shared_space_type}: {round(total_missing * 100.0/totals_expected, 2)}%")

    updated_src_data[shared_space_type] = nearest_names
    return updated_src_data



def place_agent_to_shared_space_based_on_distance(
        agent: Series,
        shared_space_loc: dict[str, DataFrame]) -> Series:
    """
    Assigns specific shared space locations to an agent based on their home area.

    For each `proc_shared_space_name` (e.g., "supermarket", "hospital") in the
    `shared_space_loc` dictionary, this function looks up the shared space(s)
    assigned to the agent's home 'area' in the corresponding DataFrame
    (e.g., `shared_space_loc['supermarket']`). It then assigns this
    shared space ID (or comma-separated IDs if multiple were found by
    `find_nearest_shared_space_from_household`) to the agent's attribute
    of the same name (e.g., `agent['supermarket']`).

    Args:
        agent (Series): The agent to be updated. Must have an 'area' attribute.
        shared_space_loc (dict[str, DataFrame]): A dictionary where keys are shared
            space types (e.g., "supermarket") and values are DataFrames. Each
            DataFrame must have an 'area' column and a column with the same name
            as the key (e.g., 'supermarket' column containing specific supermarket IDs
            assigned to that area). This dict is typically the output of multiple
            calls to `find_nearest_shared_space_from_household`.

    Returns:
        Series: The updated agent Series with specific shared space IDs assigned.
    """
    for proc_shared_space_name in shared_space_loc:
        proc_shared_space_loc = shared_space_loc[proc_shared_space_name]
        agent[proc_shared_space_name] = proc_shared_space_loc[
            proc_shared_space_loc["area"] == agent.area][proc_shared_space_name].values[0]

    return agent
