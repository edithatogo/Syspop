import logging
from json import dump as json_dump
from os import makedirs
from os.path import exists, join
from time import sleep

from funcs import RAW_DATA_DIR
from numpy import nan as numpy_nan
from numpy.random import uniform as numpy_uniform
from OSMPythonTools.nominatim import Nominatim
from OSMPythonTools.overpass import Overpass, overpassQueryBuilder
from pandas import DataFrame
from pandas import concat as pandas_concat

# Configure the logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Create a logger object
logger = logging.getLogger(__name__)


"""
from OSMPythonTools.api import Api

# Initialize the API
api = Api()

# Define the OSM ID
osm_id = 307249609

# Get the OSM object
osm_object = api.query(f'way/{osm_id}')

# Print the OSM object
print(osm_object)


"""


ADD_RANDOM_PLACES_SCALER = {
    "restaurant": 0.3,
    "supermarket": 1.5,
    "wholesale": 1.5,
    "fast_food": 3.0,
    "kindergarten": 5.0,
    "pub": 5.0,
    "cafe": 3.0,
}


QUERY_KEYS = {
    "Tonga": {
        "building": ["yes", "residential", "house"],
    },
    "New Zealand": {
        "building": ["house"],
        "amenity": [
            "restaurant",
            # "pharmacy",
            "fast_food",
            "cafe",
            # "events_venue",
            "pub",
            "gym",
            "childcare",
            "kindergarten",
            "school",
        ],
        "shop": [
            "supermarket",
            "wholesale",
            "bakery",
            "general",
            "department_store",
            "convenience",
        ],
        "leisure": ["park"],
        # "tourism": ["museum"],
    },
}


overpass = Overpass()
nominatim = Nominatim()


def add_random_location(
    df: DataFrame, total_lines: int, data_type: str, buffer: float = 0.3
) -> DataFrame:
    """
    Generates a DataFrame with randomly perturbed locations based on an input DataFrame.

    For a specified number of lines (`total_lines`), it samples a row from the
    input `df`, takes its latitude and longitude, and adds a random uniform buffer
    (between -`buffer` and +`buffer` degrees) to both.

    Args:
        df (DataFrame): Input DataFrame with 'lat' and 'lon' columns to sample from.
        total_lines (int): The number of random locations to generate.
        data_type (str): A string prefix for the 'name' of the generated locations
                         (e.g., "supermarket_pseudo_0").
        buffer (float, optional): The maximum absolute value for the random
                                  perturbation added to latitude and longitude.
                                  Defaults to 0.3.

    Returns:
        DataFrame: A new DataFrame with columns ['name', 'lat', 'lon']
                   containing the generated random locations.
    """
    output = {"name": [], "lat": [], "lon": []}

    for i in range(total_lines):
        proc_row = df.sample(n=1)
        output["name"].append(f"{data_type}_pseudo_{i}")
        output["lat"].append(proc_row.lat.values[0] + numpy_uniform(-buffer, buffer))
        output["lon"].append(proc_row.lon.values[0] + numpy_uniform(-buffer, buffer))

    return DataFrame(output)


def query_results(
    query_keys: dict,
    region: str,
    country: str,
    output_dir: str = RAW_DATA_DIR,
    if_add_random_loc: bool = False,
    use_element: bool = False,
):
    """
    Queries OpenStreetMap (OSM) using Overpass API for specified feature types
    within a given region and country, then saves the results as CSV files.

    For each key-value pair in `query_keys` (e.g., "amenity": "restaurant"):
    1. Constructs an Overpass query for nodes and ways.
    2. Executes the query.
    3. Extracts latitude, longitude, and name for each resulting element.
    4. Saves the data to a CSV file named "{key}_{value}.csv" in `output_dir`.
    5. Optionally, adds more randomly perturbed locations based on the queried data.
    6. Saves a JSON mapping of recorded pseudo-names to actual OSM names.

    Args:
        query_keys (dict): A dictionary where keys are OSM feature keys (e.g., "amenity",
                           "shop") and values are lists of feature values (e.g.,
                           ["restaurant", "cafe"], ["supermarket"]).
        region (str): The region to query within (e.g., "Wellington"). Can be numpy.nan
                      if querying the whole country.
        country (str): The country to query (e.g., "New Zealand").
        output_dir (str, optional): Directory to save the output CSV and JSON files.
                                    Defaults to RAW_DATA_DIR.
        if_add_random_loc (bool, optional): If True, adds additional randomly
            perturbed locations for certain data types (defined in
            ADD_RANDOM_PLACES_SCALER). Defaults to False.
        use_element (bool, optional): If True, processes all elements from the
            Overpass result (nodes, ways, relations). If False, only processes nodes.
            Defaults to False. (Note: Current implementation for ways might be incomplete
            if relying only on `node.lat()`, `node.lon()` for way geometries).
    """

    if not exists(output_dir):
        makedirs(output_dir)

    if region is not numpy_nan:
        areaId = nominatim.query(f"{region},{country}").areaId()
    else:
        areaId = nominatim.query(f"{country}").areaId()

    actual_and_records_name_mapping = {}
    for proc_key in query_keys:

        actual_and_records_name_mapping[proc_key] = {}

        for proc_value in query_keys[proc_key]:

            actual_and_records_name_mapping[proc_key][proc_value] = {}
            # query = overpassQueryBuilder(area=area_id, elementType='way', selector='"amenity"="school"', out='body')
            # proc_value = "school"
            query = overpassQueryBuilder(
                area=areaId,
                elementType=["node", "way"],
                selector=f'"{proc_key}"="{proc_value}"',
                out="body",
            )
            result = overpass.query(query, timeout=300)
            # result.toJSON()
            # elements = result.elements()
            # elements[0].lat() ...
            outputs = {"name": [], "lat": [], "lon": []}

            if use_element:
                all_results = result.elements()
            else:
                all_results = result.nodes()

            if all_results is None:
                continue

            for i, node in enumerate(all_results):

                logger.info(f"{proc_value}: {i} / {len(all_results)}")

                try:
                    recorded_name = f"{proc_value}_{i}"

                    proc_lat = node.lat()
                    proc_lon = node.lon()
                    if (proc_lat is None) or (proc_lon is None):
                        proc_lat = node.nodes()[0].lat()
                        proc_lon = node.nodes()[0].lon()
                except Exception:
                    sleep(10)
                    logger.info(f"Not able to connect for {i}")
                    continue

                outputs["name"].append(recorded_name)
                outputs["lat"].append(proc_lat)
                outputs["lon"].append(proc_lon)

                # print(f"{node.lat()}, {node.lon()}")

                try:
                    actual_name = node.tags()["name"]
                except KeyError:
                    actual_name = "Unknown"

                try:
                    parent_name = node.tags()["is_in"]
                    actual_name += f", {parent_name}"
                except KeyError:
                    pass

                actual_and_records_name_mapping[proc_key][proc_value][
                    recorded_name
                ] = actual_name

            df = DataFrame.from_dict(outputs)

            if if_add_random_loc and proc_value in ADD_RANDOM_PLACES_SCALER:
                random_df = add_random_location(
                    df,
                    int((ADD_RANDOM_PLACES_SCALER[proc_value] + 1.0) * len(df)),
                    proc_value,
                )
                df = pandas_concat([df, random_df], axis=0)

            df.to_csv(join(output_dir, f"{proc_key}_{proc_value}.csv"), index=False)

    with open(
        join(output_dir, "actual_and_records_name_mapping.json"), "w"
    ) as json_fid:
        json_dump(actual_and_records_name_mapping, json_fid)


if __name__ == "__main__":
    region = "Wellington"  # can be NaN, or sth like Auckland
    country = "New Zealand"  # New Zealand, Tonga
    query_results(
        QUERY_KEYS[country],
        region,
        country,
        output_dir="etc/data/raw_wellington_latest",
        if_add_random_loc=False,
        use_element=True,
    )

    # query_results(
    #    {"amenity": ["kindergarten"]}, "Wellington", country, output_dir="/tmp/test"
    # )
