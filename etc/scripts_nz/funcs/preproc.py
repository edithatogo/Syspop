
from geopandas import read_file as gpd_read_file
from geopandas import sjoin as gpd_sjoin
from pandas import DataFrame, read_csv

from funcs import POPULATION_CODE, RAW_DATA_INFO, REGION_CODES, REGION_NAMES_CONVERSIONS
from funcs.utils import get_central_point


def _read_raw_household(raw_household_path: str, include_public_dwelling: bool = False) -> DataFrame:
    """
    Reads and processes a raw household CSV file to count households by
    area, number of adults, and number of children.

    The input CSV is expected to have columns like 'SA2 Code', 'Number of people',
    'Number of adults', 'Dwelling type', and 'Count'.

    Args:
        raw_household_path (str): Path to the raw household CSV data.
        include_public_dwelling (bool, optional): If True, includes dwellings
            marked as public (dwelling_type >= 2000). Defaults to False.

    Returns:
        DataFrame: A DataFrame with columns ['area', 'adults', 'children', 'value'],
                   where 'value' is the count of households with that composition.
    """
    data = read_csv(raw_household_path)

    data = data.rename(
        columns={
            "SA2 Code": "area",
            "Number of people": "people",
            "Number of adults": "adults",
            "Dwelling type": "dwelling_type",
            "Count": "value",
        })


    if not include_public_dwelling:
        data = data[data["dwelling_type"] < 2000]

    data["adults"] = data["adults"].astype(int)
    data["people"] = data["people"].astype(int)
    data["children"] = data["people"].astype(int) - data["adults"].astype(int)

    data = data.groupby(["area", "adults", "children"], as_index=False)["value"].sum()

    return data[["area", "adults", "children", "value"]]


def _read_raw_address(raw_sa2_area_path: str, raw_address_path: str) -> DataFrame:
    """
    Performs a spatial join between address points and SA2 area polygons
    to assign an SA2 area to each address.

    The input files are expected to be GeoJSON or Shapefiles readable by GeoPandas.
    SA2 area data should have an 'SA22022_V1' column for the SA2 code.
    Address data should contain point geometries.

    Args:
        raw_sa2_area_path (str): Path to the SA2 area geometry file.
        raw_address_path (str): Path to the address point geometry file.

    Returns:
        DataFrame: A DataFrame with columns ['area', 'latitude', 'longitude']
                   for each address point that falls within an SA2 area.
    """
    sa2_data = gpd_read_file(raw_sa2_area_path)
    address_data = gpd_read_file(raw_address_path)

    gdf_sa2 = sa2_data.to_crs(epsg=4326)
    gdf_address = address_data.to_crs(epsg=4326)
    gdf_sa2 = gdf_sa2[["SA22022_V1", "geometry"]]
    gdf_address = gdf_address[["geometry"]]

    combined_df = gpd_sjoin(gdf_address, gdf_sa2, how="inner", op="within")
    combined_df["lon"] = combined_df.geometry.x
    combined_df["lat"] = combined_df.geometry.y

    combined_df = combined_df.rename(
        columns={"SA22022_V1": "area", "lat": "latitude", "lon": "longitude"}
    )

    combined_df["area"] = combined_df["area"].astype(int)

    return combined_df[["area", "latitude", "longitude"]]


def _read_raw_geography_hierarchy(raw_geography_hierarchy_path: str) -> DataFrame:
    """
    Reads a raw geography hierarchy CSV file and processes it to create a
    DataFrame mapping regions, super_areas (SA3), and areas (SA2).

    The input CSV is expected to have columns 'REGC2023_code', 'SA32023_code',
    'SA32023_name', and 'SA22018_code'.
    It filters out 'Others' regions, maps region codes to names using
    REGION_NAMES_CONVERSIONS, and removes duplicated areas.

    Args:
        raw_geography_hierarchy_path (str): Path to the raw geography
                                              hierarchy CSV data.

    Returns:
        DataFrame: A DataFrame with columns ['region', 'super_area', 'area'].
    """

    def _map_codes2(code: str) -> str | None:
        """Maps regional code to region name."""
        for key, values in REGION_NAMES_CONVERSIONS.items():
            if code == key:
                return values
        return None

    data = read_csv(raw_geography_hierarchy_path)

    data = data[["REGC2023_code", "SA32023_code", "SA32023_name", "SA22018_code"]]

    data = data[~data["REGC2023_code"].isin(REGION_CODES["Others"])]

    data["REGC2023_name"] = data["REGC2023_code"].map(_map_codes2)

    data = data.rename(
        columns={
            "REGC2023_name": "region",
            "SA32023_code": "super_area",
            "SA22018_code": "area",
            "SA32023_name": "super_area_name",
        }
    ).drop_duplicates()

    data = data[["region", "super_area", "area", "super_area_name"]]

    data = data[~data["area"].duplicated(keep=False)]

    return data[["region", "super_area", "area"]]


def _read_raw_geography_location_area(raw_geography_location_path: str) -> DataFrame:
    """
    Reads a raw geography location CSV file and extracts SA2 area codes
    with their corresponding latitudes and longitudes.

    The input CSV is expected to have columns 'SA22018_V1_00' (SA2 code),
    'LATITUDE', and 'LONGITUDE'.

    Args:
        raw_geography_location_path (str): Path to the raw geography
                                              location CSV data.

    Returns:
        DataFrame: A DataFrame with columns ['area', 'latitude', 'longitude'].
    """
    data = read_csv(raw_geography_location_path)

    data = data[["SA22018_V1_00", "LATITUDE", "LONGITUDE"]]

    data = data.rename(
        columns={
            "SA22018_V1_00": "area",
            "LATITUDE": "latitude",
            "LONGITUDE": "longitude",
        }
    )

    return data


def _read_raw_travel_to_work(raw_travel_to_work_path: str, data_type: str = "work") -> DataFrame:
    """
    Reads and processes a raw travel-to-work or travel-to-school CSV file,
    extracting relevant columns and renaming them for consistency.

    The input CSV is expected to contain columns for origin SA2 code,
    destination SA2 code (either workplace or educational address), and counts
    for various travel modes.

    Args:
        raw_travel_to_work_path (str): Path to the raw travel CSV data.
        data_type (str, optional): Specifies the type of travel, either "work"
                                   or "school". Defaults to "work".

    Returns:
        DataFrame: A processed DataFrame containing travel mode counts between
                   origin and destination areas. Columns are standardized.
                   Replaces -999.0 with 0.
    """

    data = read_csv(raw_travel_to_work_path)

    if data_type == "work":
        data = data[
            [
                "SA2_code_usual_residence_address",
                "SA2_code_workplace_address",
                "Work_at_home",
                "Drive_a_private_car_truck_or_van",
                "Drive_a_company_car_truck_or_van",
                "Passenger_in_a_car_truck_van_or_company_bus",
                "Public_bus",
                "Train",
                "Bicycle",
                "Walk_or_jog",
                "Ferry",
                "Other",
            ]
        ]
        data.rename(
            columns={
                "SA2_code_usual_residence_address": "area_home",
                "SA2_code_workplace_address": "area_work",
            },
            inplace=True,
        )
    elif data_type == "school":
        data = data[
            [
                "SA2_code_usual_residence_address",
                "SA2_code_educational_address",
                "Study_at_home",
                "Drive_a_car_truck_or_van",
                "Passenger_in_a_car_truck_or_van",
                "Train",
                "Bicycle", "Walk_or_jog", "School_bus", "Public_bus", "Ferry", "Other"
            ]
        ]
        data.rename(
            columns={
                "SA2_code_usual_residence_address": "area_home",
                "SA2_code_educational_address": "area_school",
            },
            inplace=True,
        )

    data = data.replace(-999.0, 0)

    return data


def _read_raw_income_data(income_path: str) -> DataFrame:
    """
    Reads a raw income CSV file, renames the 'sex' column to 'gender',
    and replaces coded values with their string representations using POPULATION_CODE.

    Args:
        income_path (str): Path to the raw income CSV data.

    Returns:
        DataFrame: Processed income DataFrame.
    """
    data_income = read_csv(income_path).reset_index(drop=True)
    data_income = data_income.rename(columns={"sex": "gender"})
    data_income = data_income.replace(POPULATION_CODE)
    return data_income

def _read_raw_employer_employee_data(employer_employee_num_path: str) -> DataFrame:
    """
    Reads a raw CSV file containing employer and employee counts by ANZSIC06
    business code and area.

    The input CSV is expected to have columns 'anzsic06', 'Area', 'ec_count'
    (employee count), and 'geo_count' (employer count).
    It filters for top-level ANZSIC codes (single character) and areas
    starting with 'A', then renames columns for clarity.

    Args:
        employer_employee_num_path (str): Path to the raw employer/employee
                                           count CSV data.

    Returns:
        DataFrame: A DataFrame with columns ['area', 'business_code',
                   'employer', 'employee'].
    """

    data = read_csv(
        employer_employee_num_path)[["anzsic06", "Area", "ec_count", "geo_count"]]

    data = data[data["Area"].str.startswith("A")]
    data = data[data["anzsic06"].apply(lambda x: len(x) == 1)]

    data = data.rename(columns={"Area": "area"})

    data["area"] = data["area"].str[1:].astype(int)

    data = data.rename(columns={
        "anzsic06": "business_code",
        "ec_count": "employee",
        "geo_count": "employer"})

    return data[[
        "area",
        "business_code",
        "employer",
        "employee"]
    ]


def _read_raw_schools(school_data_path: str) -> DataFrame:
    """
    Reads and processes a raw school data CSV file.

    Filters for entries where 'use' is 'School', excludes specific 'use_type'
    categories (e.g., "Teen Parent Unit"), maps 'use_type' to a standardized
    sector and age range using `RAW_DATA_INFO`, and extracts coordinates.

    Args:
        school_data_path (str): Path to the raw school CSV data. The CSV should
                                contain 'use', 'use_type', 'WKT' (Well-Known Text
                                for geometry), and 'estimated_occupancy' columns.

    Returns:
        DataFrame: A DataFrame with columns ['estimated_occupancy', 'age_min',
                   'age_max', 'latitude', 'longitude', 'sector'].
    """

    data = read_csv(school_data_path)

    data = data[data["use"] == "School"]

    data = data[
        ~data["use_type"].isin(
            [
                "Teen Parent Unit",
                "Correspondence School",
            ]
        )
    ]

    data["use_type"] = data["use_type"].map(
        RAW_DATA_INFO["base"]["venue"]["school"]["school_age_table"]
    )

    data[["sector", "age_range"]] = data["use_type"].str.split(" ", n=1, expand=True)
    data["age_range"] = data["age_range"].str.strip("()")
    data[["age_min", "age_max"]] = data["age_range"].str.split("-", expand=True)

    # data[["sector", "age_min", "age_max"]] = data["use_type"].str.extract(
    #    r"([A-Za-z\s]+)\s\((\d+)-(\d+)\)"
    # )

    data["Central Point"] = data["WKT"].apply(get_central_point)

    data["latitude"] = data["Central Point"].apply(lambda point: point.y)
    data["longitude"] = data["Central Point"].apply(lambda point: point.x)

    return data[["estimated_occupancy", "age_min", "age_max", "latitude", "longitude", "sector"]]


def _read_raw_kindergarten(raw_kindergarten_path: str) -> DataFrame:
    """
    Reads and processes a raw kindergarten CSV file.

    Filters for kindergartens with more than 15 licensed positions.
    Extracts 'Statistical Area 2 Code', 'Max. Licenced Positions', 'Latitude',
    and 'Longitude'. Renames columns and adds 'sector', 'age_min', 'age_max'.

    Args:
        raw_kindergarten_path (str): Path to the raw kindergarten CSV data.

    Returns:
        DataFrame: A DataFrame with columns ['area', 'max_students', 'latitude',
                   'longitude', 'sector', 'age_min', 'age_max'].
    """
    df = read_csv(raw_kindergarten_path)

    df = df[df["Max. Licenced Positions"] > 15.0]

    df = df[
        [
            "Statistical Area 2 Code",
            "Max. Licenced Positions",
            "Latitude",
            "Longitude",
        ]
    ]

    df = df.rename(
        columns={
            "Statistical Area 2 Code": "area",
            "Max. Licenced Positions": "max_students",
            "Latitude": "latitude",
            "Longitude": "longitude",
        }
    )
    df = df.dropna()

    df["area"] = df["area"].astype(int)
    df["max_students"] = df["max_students"].astype(int)

    df["sector"] = "kindergarten"
    df["age_min"] = 0
    df["age_max"] = 5

    return df


def _read_raw_hospital(raw_hospital_data_path: str) -> DataFrame:
    """
    Reads and processes a raw hospital data CSV file.

    Filters for entries where 'use' is 'Hospital'. Extracts coordinates from
    'WKT' (Well-Known Text) and selects relevant columns.

    Args:
        raw_hospital_data_path (str): Path to the raw hospital CSV data.
                                      Expected to have 'use', 'WKT',
                                      'estimated_occupancy', and
                                      'source_facility_id' columns.

    Returns:
        DataFrame: A DataFrame with columns ['latitude', 'longitude',
                   'estimated_occupancy', 'source_facility_id'].
    """
    data = read_csv(raw_hospital_data_path)

    data = data[data["use"] == "Hospital"]

    data["Central Point"] = data["WKT"].apply(get_central_point)

    data["latitude"] = data["Central Point"].apply(lambda point: point.y)
    data["longitude"] = data["Central Point"].apply(lambda point: point.x)

    return data[["latitude", "longitude", "estimated_occupancy", "source_facility_id"]]

def _read_original_csv(osm_data_path: str) -> DataFrame:
    """
    Reads a CSV file from the given path and removes duplicate rows.

    Args:
        osm_data_path (str): The file path to the CSV data, typically
                             OpenStreetMap data.

    Returns:
        DataFrame: A DataFrame with duplicate rows removed.
    """
    return read_csv(osm_data_path).drop_duplicates()
