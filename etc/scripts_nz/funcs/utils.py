from copy import deepcopy

from numpy import arctan2, cos, nan, radians, sin, sqrt
from pandas import DataFrame, read_csv, read_excel
from shapely.wkt import loads as wkt_loads
from yaml import safe_load


def sort_column_by_names(data_input: DataFrame, columns_to_exclude: list) -> DataFrame:
    """
    Sorts the columns of a DataFrame, typically by numeric age groups,
    while keeping specified columns at the beginning.

    Args:
        data_input (DataFrame): The input DataFrame.
        columns_to_exclude (list): A list of column names to keep at the
                                   beginning of the DataFrame, in the order
                                   they appear in this list. Other columns
                                   are assumed to be sortable as integers.

    Returns:
        DataFrame: The DataFrame with columns reordered.
    """
    cols = list(data_input.columns)

    # Remove 'output_area' and 'gender' from the list
    cols_excluded = []
    for proc_col_to_remove in columns_to_exclude:
        if proc_col_to_remove in cols:
            cols.remove(proc_col_to_remove)
            cols_excluded.append(proc_col_to_remove)

    # Sort the remaining columns
    cols = sorted(cols, key=lambda x: int(x))

    # Add 'output_area' and 'gender' at the start
    cols = cols_excluded + cols

    # Reindex the dataframe
    return data_input.reindex(columns=cols)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculates the Haversine distance between two geographical points.

    Args:
        lat1 (float): Latitude of the first point in degrees.
        lon1 (float): Longitude of the first point in degrees.
        lat2 (float): Latitude of the second point in degrees.
        lon2 (float): Longitude of the second point in degrees.

    Returns:
        float: The distance between the two points in kilometers.
    """
    r = 6371  # Earth's radius in kilometers
    phi1 = radians(lat1)
    phi2 = radians(lat2)
    delta_phi = radians(lat2 - lat1)
    delta_lambda = radians(lon2 - lon1)
    a = sin(delta_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(delta_lambda / 2) ** 2
    c = 2 * arctan2(sqrt(a), sqrt(1 - a))
    distance = r * c
    return distance


def get_central_point(wkt_string: str) -> wkt_loads:
    """
    Calculates the centroid of a geometry represented by a WKT (Well-Known Text) string.

    Args:
        wkt_string (str): The Well-Known Text representation of a geometry
                          (e.g., "POLYGON((...))", "MULTIPOLYGON(((...)))").

    Returns:
        shapely.geometry.point.Point: The centroid of the input geometry.
    """
    polygon = wkt_loads(wkt_string)
    central_point = polygon.centroid
    return central_point


def read_anzsic_code(anzsic06_code_path: str) -> DataFrame:
    """
    Reads an ANZSIC06 code CSV file and cleans the 'Description' column.

    The 'Description' column is expected to have a leading code/number
    followed by the actual description (e.g., "A Agriculture, Forestry and Fishing").
    This function removes the leading code/number part.

    Args:
        anzsic06_code_path (str): Path to the ANZSIC06 code CSV file.

    Returns:
        DataFrame: The processed DataFrame with cleaned 'Description'.
    """
    anzsic_code = read_csv(anzsic06_code_path)

    for _, row in anzsic_code.iterrows():
        row["Description"] = " ".join(row["Description"].split()[1:])

    return anzsic_code


def read_leed(
    leed_path: str, anzsic_code: DataFrame | None, if_rate: bool = False
) -> DataFrame:
    """
    Reads and processes New Zealand Longitudinal Employer-Employee Data (LEED).

    This function reshapes the LEED Excel file, maps industry descriptions to
    ANZSIC06 codes (if `anzsic_code` DataFrame is provided), aggregates data to
    top-level ANZSIC codes, and optionally calculates employment rates by gender.

    Args:
        leed_path (str): Path to the LEED Excel file.
        anzsic_code (DataFrame | None): A DataFrame mapping ANZSIC06 descriptions
            to codes. If None, industry codes are not mapped.
        if_rate (bool, optional): If True, calculates and returns employment rates
            by gender for each industry and area. Otherwise, returns counts.
            Defaults to False.

    Returns:
        DataFrame: Processed LEED data. If `if_rate` is True, returns rates;
                   otherwise, returns counts by Area, Age, and Industry (Male/Female).
    """
    df = read_excel(leed_path)
    industrial_row = df.iloc[0].fillna(method="ffill")

    if anzsic_code is not None:
        for i, row in enumerate(industrial_row):
            row = row.strip()

            if row in ["Industry", "Total people"]:
                continue

            code = anzsic_code[anzsic_code["Description"] == row]["Anzsic06"].values[0]
            industrial_row[i] = code

    # x = anzsic_code.set_index("Description")
    sec_row = df.iloc[1].fillna(method="ffill")
    titles = industrial_row + "," + sec_row
    titles[
        "Number of Employees by Industry, Age Group, Sex, and Region (derived from 2018 Census)"
    ] = "Area"
    titles["Unnamed: 1"] = "Age"
    titles["Unnamed: 2"] = "tmp"

    df = df.iloc[3:]
    df = df.drop(df.index[-1:])
    df = df.rename(columns=titles)
    df = df.drop("tmp", axis=1)
    df["Area"] = df["Area"].fillna(method="ffill")
    # return df.rename(columns=lambda x: x.strip())

    df["Area"] = df["Area"].replace(
        "Manawatu-Wanganui Region", "Manawatu-Whanganui Region"
    )

    if anzsic_code is not None:
        character_indices = set(
            [
                col.split(",")[0][0]
                for col in df.columns
                if col
                not in ["Area", "Age", "Total people,Male", "Total people, Female"]
            ]
        )

        # Iterate over the unique character indices to sum the corresponding columns
        for char_index in character_indices:
            subset_cols_male = [
                col
                for col in df.columns
                if col.startswith(char_index)
                and col.endswith("Male")
                and col
                not in ["Area", "Age", "Total people,Male", "Total people,Female"]
            ]
            subset_cols_female = [
                col
                for col in df.columns
                if col.startswith(char_index)
                and col.endswith("Female")
                and col
                not in ["Area", "Age", "Total people,Male", "Total people,Female"]
            ]
            summed_col_male = f"{char_index},Male"
            summed_col_female = f"{char_index},Female"
            df[summed_col_male] = df[subset_cols_male].sum(axis=1)
            df[summed_col_female] = df[subset_cols_female].sum(axis=1)
            df = df.drop(subset_cols_male + subset_cols_female, axis=1)

    df["Area"] = df["Area"].str.replace(" Region", "")

    if not if_rate:
        return df

    industrial_columns = [
        x
        for x in list(df.columns)
        if x not in ["Area", "Age", "Total people,Male", "Total people,Female"]
    ]

    df = df.groupby("Area")[industrial_columns].sum()

    df_rate = deepcopy(df)

    # Calculate percentages
    for column in df.columns:
        group = column.split(",")[0]
        total = df[[f"{group},Male", f"{group},Female"]].sum(
            axis=1
        )  # Calculate the total for the group

        total.replace(0, nan, inplace=True)
        df_rate[column] = df[column] / total

    return df_rate


def read_cfg(cfg_path: str) -> dict:
    """
    Reads a YAML configuration file.

    Args:
        cfg_path (str): Path to the YAML configuration file.

    Returns:
        dict: The configuration loaded from the YAML file.
    """
    with open(cfg_path) as fid:
        cfg = safe_load(fid)

    return cfg
