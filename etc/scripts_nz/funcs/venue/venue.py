
from numpy import argmin
from pandas import DataFrame
from scipy.spatial.distance import cdist

from funcs.preproc import _read_original_csv, _read_raw_hospital, _read_raw_kindergarten, _read_raw_schools
from funcs.utils import haversine_distance


def create_osm_space(data_path: str, geography_location: DataFrame) -> DataFrame:
    """
    Processes OpenStreetMap (OSM) data for shared spaces, assigning an SA2 area
    to each space based on proximity to SA2 centroids.

    Reads OSM data from a CSV (expected to have 'lat', 'lon' columns),
    calculates the Haversine distance to SA2 area centroids, and assigns
    the area of the nearest centroid.

    Args:
        data_path (str): Path to the CSV file containing OSM shared space data
                         (e.g., supermarkets, restaurants).
        geography_location (DataFrame): DataFrame with SA2 area centroids,
                                        containing 'area', 'latitude', 'longitude'.

    Returns:
        DataFrame: The input OSM data DataFrame with an added 'area' column
                   and renamed 'lat'/'lon' to 'latitude'/'longitude'.
                   Rows with no assigned area (due to no nearby centroid)
                   are dropped.
    """
    data = _read_original_csv(data_path)
    distances = cdist(
        data[["lat", "lon"]],
        geography_location[["latitude", "longitude"]],
        lambda x, y: haversine_distance(x[0], x[1], y[0], y[1]),
    )
    # Find the nearest location in A for each point in B
    nearest_indices = argmin(distances, axis=1)
    data["area"] = geography_location["area"].iloc[nearest_indices].values

    data.dropna(inplace=True)
    data[["area"]] = data[["area"]].astype(int)
    data = data.rename(columns={"lat": "latitude", "lon": "longitude"})

    return data


def create_kindergarten(kindergarten_data_path: str) -> DataFrame:
    """
    Reads and processes raw New Zealand kindergarten data by calling
    `_read_raw_kindergarten`.

    Args:
        kindergarten_data_path (str): The file path to the CSV file
                                      containing the raw kindergarten data.

    Returns:
        DataFrame: A pandas DataFrame containing the processed kindergarten data,
                   as returned by `_read_raw_kindergarten`.
    """
    return _read_raw_kindergarten(kindergarten_data_path)


def create_school(
    school_data_path: str,
    sa2_loc: DataFrame,
    max_to_cur_occupancy_ratio=1.2,
) -> DataFrame:
    """
    Processes raw school data, assigns SA2 areas based on proximity,
    and calculates maximum student capacity.

    Calls `_read_raw_schools` to get initial school data. Then, for each school,
    it finds the nearest SA2 area centroid from `sa2_loc` and assigns that
    SA2 area to the school. Maximum student capacity is estimated by multiplying
    'estimated_occupancy' by `max_to_cur_occupancy_ratio`.

    Args:
        school_data_path (str): Path to the raw school CSV data.
        sa2_loc (DataFrame): DataFrame with SA2 area centroids, containing
                             'area', 'latitude', 'longitude'.
        max_to_cur_occupancy_ratio (float, optional): Factor to multiply
            estimated occupancy by to get maximum capacity. Defaults to 1.2.

    Returns:
        DataFrame: A DataFrame with columns ['area', 'max_students', 'sector',
                   'latitude', 'longitude', 'age_min', 'age_max'].
    """
    data = _read_raw_schools(school_data_path)

    distances = cdist(
        data[["latitude", "longitude"]],
        sa2_loc[["latitude", "longitude"]],
        lambda x, y: haversine_distance(x[0], x[1], y[0], y[1]),
    )

    # Find the nearest location in A for each point in B
    nearest_indices = argmin(distances, axis=1)
    data["area"] = sa2_loc["area"].iloc[nearest_indices].values

    data["max_students"] = data["estimated_occupancy"] * max_to_cur_occupancy_ratio

    data["max_students"] = data["max_students"].astype(int)

    data = data[
        [
            "area",
            "max_students",
            "sector",
            "latitude",
            "longitude",
            "age_min",
            "age_max",
        ]
    ]

    # make sure columns are in integer
    for proc_key in ["area", "max_students", "age_min", "age_max"]:
        data[proc_key] = data[proc_key].astype(int)

    # make sure columns are in float
    for proc_key in ["latitude", "longitude"]:
        data[proc_key] = data[proc_key].astype(float)

    return data


def create_hospital(
    hospital_data_path: str,
    sa2_loc: DataFrame,
) -> DataFrame:
    """
    Processes raw hospital data, assigns SA2 areas based on proximity,
    and renames 'estimated_occupancy' to 'beds'.

    Calls `_read_raw_hospital` to get initial hospital data. Then, for each
    hospital, it finds the nearest SA2 area centroid from `sa2_loc` and
    assigns that SA2 area to the hospital.

    Args:
        hospital_data_path (str): Path to the raw hospital CSV data.
        sa2_loc (DataFrame): DataFrame with SA2 area centroids, containing
                             'area', 'latitude', 'longitude'.

    Returns:
        DataFrame: A DataFrame with columns ['area', 'latitude', 'longitude', 'beds'].
                   Rows with missing data after processing are dropped.
    """
    data = _read_raw_hospital(hospital_data_path)

    distances = cdist(
        data[["latitude", "longitude"]],
        sa2_loc[["latitude", "longitude"]],
        lambda x, y: haversine_distance(x[0], x[1], y[0], y[1]),
    )

    # Find the nearest location in A for each point in B
    nearest_indices = argmin(distances, axis=1)
    data["area"] = sa2_loc["area"].iloc[nearest_indices].values

    data.drop(columns=["source_facility_id"], inplace=True)

    data = data.rename(
        columns={
            "estimated_occupancy": "beds",
        }
    )
    data.dropna(inplace=True)
    data[["beds", "area"]] = data[["beds", "area"]].astype(int)

    return data[["area", "latitude", "longitude", "beds"]]
