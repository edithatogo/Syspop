

from funcs.preproc import _read_raw_household


def create_household_and_dwelling_number(raw_household_path: str) -> DataFrame:
    """
    Creates a DataFrame representing household composition and dwelling numbers
    by processing a raw household data file.

    This function acts as a wrapper around `_read_raw_household` from the
    `funcs.preproc` module.

    Args:
        raw_household_path (str): The file path to the raw household CSV data.
                                  This data is expected to be in a format that
                                  `_read_raw_household` can process.

    Returns:
        DataFrame: A DataFrame containing processed household data, typically
                   with columns like 'area', 'adults', 'children', and 'value'
                   (representing the count of such households).
    """
    return _read_raw_household(raw_household_path)

"""
def create_household_number(workdir):

    data = read_csv(RAW_DATA["household"]["household_number"])

    data = data[
        ["SA2 Code", "People Count Code", "Dependent Children Count Code", "Count"]
    ]

    data["Dependent Children Count"] = data["Dependent Children Count Code"].map(
        DEPENDENT_CHILDREN_COUNT_CODE
    )
    data["Dependent Children Count"] = (
        data["Dependent Children Count"]
        .apply(lambda x: random_randint(0, 5) if pandas_isnull(x) else x)
        .astype(int)
    )

    data["Count"] = (
        data["Count"]
        .apply(lambda x: random_randint(1, 5) if x == "s" else x)
        .astype(int)
    )

    data = data.rename(
        columns={
            "SA2 Code": "area",
            "People Count Code": "people_num",
            "Dependent Children Count": "children_num",
            "Count": "household_num",
        }
    )

    data = data[["area", "people_num", "children_num", "household_num"]]

    # remove duplicated household composition
    data = data.groupby(["area", "people_num", "children_num"], as_index=False)[
        "household_num"
    ].sum()

    return data
"""
