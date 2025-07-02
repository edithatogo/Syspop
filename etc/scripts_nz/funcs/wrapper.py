from logging import getLogger
from os.path import join

from pandas import read_parquet

from funcs.household.household import create_household_and_dwelling_number
from funcs.others.health import add_mmr
from funcs.others.immigration import add_birthplace
from funcs.population.population import read_population_structure
from funcs.preproc import (
    _read_raw_address,
    _read_raw_employer_employee_data,
    _read_raw_geography_hierarchy,
    _read_raw_geography_location_area,
    _read_raw_income_data,
    _read_raw_travel_to_work,
)
from funcs.venue.venue import create_hospital, create_kindergarten, create_osm_space, create_school

logger = getLogger()


def create_household_wrapper(workdir: str, input_cfg: dict):
    """
    Wrapper function to create and save household composition data.

    Calls `create_household_and_dwelling_number` to generate household data
    based on the 'household_composition' path in `input_cfg`.
    Saves the resulting DataFrame as "household_composition.parquet" in `workdir`.

    Args:
        workdir (str): Directory to save the output Parquet file.
        input_cfg (dict): Configuration dictionary containing the path to
                          the raw household composition data under
                          `input_cfg['household']['household_composition']`.
    """

    hhd_data = create_household_and_dwelling_number(
        input_cfg["household"]["household_composition"])
    # hhd_data['percentage'] = hhd_data.groupby("area")["num"].transform(
    #    lambda x: x / x.sum())
    hhd_data.to_parquet(join(
        workdir, "household_composition.parquet"))


def create_shared_space_wrapper(
    workdir: str, input_cfg: dict, space_names: list = ["supermarket", "restaurant", "pharmacy"]
):
    """
    Wrapper function to create and save data for various types of shared spaces.

    For each `space_name` in `space_names`:
    1. Reads the geography location data ("geography_location.parquet").
    2. Calls `create_osm_space` to process the OSM data for that shared space type,
       using the path specified in `input_cfg['venue'][space_name]`.
    3. Saves the resulting DataFrame as "shared_space_{space_name}.parquet" in `workdir`.

    Args:
        workdir (str): Directory to save the output Parquet files.
        input_cfg (dict): Configuration dictionary containing paths to OSM data
                          for each shared space type under `input_cfg['venue']`.
        space_names (list, optional): A list of shared space types to process.
            Defaults to ["supermarket", "restaurant", "pharmacy"].
    """
    geography_location = read_parquet(join(workdir, "geography_location.parquet"))

    for space_name in space_names:
        print(f"Shared space: {space_name} ...")
        proc_data = create_osm_space(input_cfg["venue"][space_name], geography_location)
        proc_data.to_parquet(join(workdir, f"shared_space_{space_name}.parquet"))


def create_others_wrapper(workdir: str, input_cfg: dict):
    """
    Wrapper function to create and save miscellaneous datasets like MMR vaccine
    and birthplace data.

    Calls `add_mmr` and `add_birthplace` using paths from `input_cfg['others']`.
    Saves the resulting DataFrames as "mmr_vaccine.parquet" and
    "birthplace.parquet" in `workdir`.

    Args:
        workdir (str): Directory to save the output Parquet files.
        input_cfg (dict): Configuration dictionary containing paths to raw data
                          for MMR vaccine and birthplace under `input_cfg['others']`.
    """
    mmr_data = add_mmr(input_cfg["others"]["mmr_vaccine"])
    birthplace_data = add_birthplace(input_cfg["others"]["birthplace"])
    mmr_data.to_parquet(join(workdir, "mmr_vaccine.parquet"))
    birthplace_data.to_parquet(join(workdir, "birthplace.parquet"))


def create_hospital_wrapper(workdir: str, input_cfg: dict):
    """
    Wrapper function to create and save hospital location data.

    1. Reads geography location data ("geography_location.parquet").
    2. Calls `create_hospital` to process raw hospital data (path from
       `input_cfg['venue']['hospital']`) and assign SA2 areas.
    3. Saves the resulting DataFrame as "hospital.parquet" in `workdir`.

    Args:
        workdir (str): Directory to save the output Parquet file.
        input_cfg (dict): Configuration dictionary containing the path to
                          raw hospital data under `input_cfg['venue']['hospital']`.
    """
    geography_location = read_parquet(join(workdir, "geography_location.parquet"))
    hopital_data = create_hospital(input_cfg["venue"]["hospital"], geography_location)
    hopital_data.to_parquet(join(workdir, "hospital.parquet"))


def create_kindergarten_wrapper(workdir: str, input_cfg: dict):
    """
    Wrapper function to create and save kindergarten location data.

    Calls `create_kindergarten` to process raw kindergarten data (path from
    `input_cfg['venue']['kindergarten']`).
    Saves the resulting DataFrame as "kindergarten.parquet" in `workdir`.

    Args:
        workdir (str): Directory to save the output Parquet file.
        input_cfg (dict): Configuration dictionary containing the path to
                          raw kindergarten data under `input_cfg['venue']['kindergarten']`.
    """
    kindergarten_data = create_kindergarten(input_cfg["venue"]["kindergarten"])
    kindergarten_data.to_parquet(join(workdir, "kindergarten.parquet"))


def create_school_wrapper(workdir: str, input_cfg: dict):
    """
    Wrapper function to create and save school location and attribute data.

    1. Reads geography location data ("geography_location.parquet").
    2. Calls `create_school` to process raw school data (path from
       `input_cfg['venue']['school']`) and assign SA2 areas.
    3. Saves the resulting DataFrame as "school.parquet" in `workdir`.

    Args:
        workdir (str): Directory to save the output Parquet file.
        input_cfg (dict): Configuration dictionary containing the path to
                          raw school data under `input_cfg['venue']['school']`.
    """

    geography_location = read_parquet(join(workdir, "geography_location.parquet"))

    school_data = create_school(input_cfg["venue"]["school"], geography_location)

    school_data.to_parquet(join(workdir, "school.parquet"))


def create_work_wrapper(workdir: str, input_cfg: dict):
    """
    Wrapper function to create and save work-related datasets: employee counts,
    employer counts, and income data.

    1. Calls `_read_raw_employer_employee_data` using the path from
       `input_cfg["work"]["employer_employee_num"]`.
    2. Calls `_read_raw_income_data` using the path from `input_cfg["work"]["income"]`.
    3. Processes the employer/employee data to create separate DataFrames for
       employee counts and employer counts.
    4. Saves "work_employee.parquet", "work_employer.parquet", and
       "work_income.parquet" in `workdir`.

    Args:
        workdir (str): Directory to save the output Parquet files.
        input_cfg (dict): Configuration dictionary containing paths to raw data for
                          employer/employee numbers and income.
    """
    data = _read_raw_employer_employee_data(input_cfg["work"]["employer_employee_num"])

    data_income = _read_raw_income_data(input_cfg["work"]["income"])

    employee_data = data[
        [
            "area",
            "business_code",
            "employee"
        ]
    ].reset_index(drop=True)

    employer_data = data[
        ["area", "business_code", "employer"]
    ].reset_index(drop=True)

    employee_data.to_parquet(join(workdir, "work_employee.parquet"))
    employer_data.to_parquet(join(workdir, "work_employer.parquet"))
    data_income.to_parquet(join(workdir, "work_income.parquet"))


def create_travel_wrapper(workdir: str, input_cfg: dict):
    """
    Wrapper function to create and save travel-to-work and travel-to-school datasets.

    Calls `_read_raw_travel_to_work` for both "work" and "school" data types,
    using paths from `input_cfg["commute"]`.
    Saves the resulting DataFrames as "commute_travel_to_work.parquet" and
    "commute_travel_to_school.parquet" in `workdir`.

    Args:
        workdir (str): Directory to save the output Parquet files.
        input_cfg (dict): Configuration dictionary containing paths to raw
                          travel data under `input_cfg["commute"]`.
    """

    trave_to_work_data = _read_raw_travel_to_work(
        input_cfg["commute"]["travel_to_work"], data_type="work")

    trave_to_school_data = _read_raw_travel_to_work(
        input_cfg["commute"]["travel_to_school"], data_type="school")

    trave_to_work_data.to_parquet(join(workdir, "commute_travel_to_work.parquet"))
    trave_to_school_data.to_parquet(join(workdir, "commute_travel_to_school.parquet"))


def create_population_wrapper(workdir: str, input_cfg: dict):
    """
    Wrapper function to create and save population structure data.

    Calls `read_population_structure` to process raw population data (path
    from `input_cfg["population"]["population_structure"]`).
    Saves the resulting DataFrame as "population_structure.parquet" in `workdir`.

    Args:
        workdir (str): Directory to save the output Parquet file.
        input_cfg (dict): Configuration dictionary containing the path to raw
                          population structure data.
    """
    population_structure = read_population_structure(
        input_cfg["population"]["population_structure"])
    population_structure.to_parquet(
        join(workdir, "population_structure.parquet"))


def create_geography_wrapper(workdir: str, input_cfg: dict, include_address: bool = True):
    """
    Wrapper function to create and save geography-related datasets: hierarchy,
    area locations, and (optionally) address points.

    - Calls `_read_raw_geography_hierarchy` for hierarchy data.
    - Calls `_read_raw_geography_location_area` for SA2 area centroid locations.
    - If `include_address` is True, calls `_read_raw_address` for address points.
    Saves the resulting DataFrames as "geography_hierarchy.parquet",
    "geography_location.parquet", and (if applicable) "geography_address.parquet"
    in `workdir`.

    Args:
        workdir (str): Directory to save the output Parquet files.
        input_cfg (dict): Configuration dictionary containing paths to raw
                          geography data.
        include_address (bool, optional): Whether to process and include
                                          address-level data. Defaults to True.
    """
    output= {
        "hierarchy": _read_raw_geography_hierarchy(
            input_cfg["geography"]["geography_hierarchy"]),
        "location": _read_raw_geography_location_area(
            input_cfg["geography"]["geography_location"])
        }
    if include_address:
        output["address"] = _read_raw_address(
            input_cfg["geography"]["sa2_area_data"],
            input_cfg["geography"]["address_data"])

    output["hierarchy"].to_parquet(join(workdir, "geography_hierarchy.parquet"))
    output["location"].to_parquet(join(workdir, "geography_location.parquet"))
    output["address"].to_parquet(join(workdir, "geography_address.parquet"))

