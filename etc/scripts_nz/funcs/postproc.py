from copy import copy as shallow_copy
from os.path import join

from numpy import unique

from funcs import AREAS_CONSISTENCY_CHECK


def postproc(
    workdir: str,
    data_list: list,
    scale: float = 1.0,
    domains_cfg: dict or None = None,
    pop=None,
):
    """
    Post-processes a list of DataFrames, standardizing them based on common
    geographical areas and an optional scaling factor.

    The function performs several steps:
    1. Scales the 'age_data' DataFrame by the given `scale` factor.
    2. Filters 'age_data' to remove areas with no people or no people over 18.
    3. Identifies common geographical areas (super_area, area) present across
       all relevant DataFrames in `data_list`.
    4. Optionally filters these common areas based on `domains_cfg` (region,
       super_area, area).
    5. Filters each DataFrame in `data_list` to include only the determined
       common geographical areas.
    6. Writes the processed DataFrames to CSV files in the `workdir`.

    Args:
        workdir (str): The directory to save the processed CSV files.
        data_list (dict[str, DataFrame]): A dictionary where keys are data names
            (e.g., "age_data", "geography_hierarchy_data") and values are the
            corresponding DataFrames. These DataFrames are expected to have area
            columns that match keys in `AREAS_CONSISTENCY_CHECK`.
        scale (float, optional): A factor to scale population counts in 'age_data'.
            Defaults to 1.0.
        domains_cfg (dict | None, optional): A dictionary to filter areas by
            specific regions, super_areas, or areas. Keys can be "region",
            "super_area", "area", with values being lists of names/IDs to keep.
            Defaults to None (no domain filtering).
        pop (Any, optional): Unused parameter. Defaults to None.
    """

    def _find_common_values(sublists):
        # Initialize with the first sublist
        common_values = set(sublists[0])

        # Iterate over the remaining sublists
        for sublist in sublists[1:]:
            common_values = common_values.intersection(sublist)

        return common_values

    # scaling the population
    age_profile = data_list["age_data"]
    columns_to_multiply = [
        col for col in age_profile.columns if col not in ["output_area"]
    ]
    age_profile[columns_to_multiply] = age_profile[columns_to_multiply] * scale

    age_profile[columns_to_multiply] = age_profile[columns_to_multiply].astype(int)
    # age_profile[columns_to_multiply] = age_profile[columns_to_multiply].applymap(math_ceil)

    # total_person = age_profile[columns_to_multiply].values.sum()
    # age_profile[columns_to_multiply] = age_profile[columns_to_multiply].astype(int)

    # remove areas with no people live
    age_profile["sum"] = age_profile[columns_to_multiply].sum(axis=1)
    age_profile = age_profile[age_profile["sum"] != 0]

    # remove areas with no people > 18
    cols_to_sum = list(range(18, 101))
    age_profile["sum18"] = age_profile[cols_to_sum].sum(axis=1)
    age_profile = age_profile[age_profile["sum18"] != 0]

    data_list["age_data"] = age_profile.drop(["sum", "sum18"], axis=1)

    # get all super_areas/areas:
    all_geo = {"super_area": [], "area": []}
    for data_name in data_list:
        if AREAS_CONSISTENCY_CHECK[data_name] is None:
            continue

        proc_data = data_list[data_name]

        for area_key in all_geo:
            if area_key in AREAS_CONSISTENCY_CHECK[data_name]:
                data_key = AREAS_CONSISTENCY_CHECK[data_name][area_key]

                all_geo[area_key].append(
                    [int(item) for item in list(unique(proc_data[data_key].values))]
                )

    for area_key in all_geo:
        all_geo[area_key] = _find_common_values(all_geo[area_key])

    # remove region, super_area or area as required
    if domains_cfg is not None:
        # domains_cfg = {"region": ["Auckland"], "super_area": None, "area": None}
        geography_hierarchy_definition = shallow_copy(
            data_list["geography_hierarchy_data"]["data"]
        )
        for domain_key in ["region", "super_area", "area"]:
            proc_domains = domains_cfg[domain_key]

            if proc_domains is not None:
                geography_hierarchy_definition = geography_hierarchy_definition[
                    geography_hierarchy_definition[domain_key].isin(proc_domains)
                ]

            for area_key in all_geo:
                all_geo[area_key] = all_geo[area_key].intersection(
                    list(geography_hierarchy_definition[area_key].unique())
                )

    # all_geo = remove_super_area(exclude_super_areas, data_list, all_geo)

    # extract data with overlapped areas
    for data_name in data_list:
        if AREAS_CONSISTENCY_CHECK[data_name] is None:
            continue

        proc_data = data_list[data_name]

        for area_key in ["super_area", "area"]:
            if area_key in AREAS_CONSISTENCY_CHECK[data_name]:
                data_key = AREAS_CONSISTENCY_CHECK[data_name][area_key]

                proc_data[data_key] = proc_data[data_key].astype(int)

                if isinstance(data_key, str):
                    proc_data = proc_data[proc_data[data_key].isin(all_geo[area_key])]
                elif isinstance(data_key, list):
                    if len(data_key) == 2:
                        proc_data = proc_data[
                            proc_data[data_key[0]].isin(all_geo[area_key])
                            & proc_data[data_key[1]].isin(all_geo[area_key])
                        ]
                    else:
                        raise Exception("does not support data_key > 2 at the moment")

        data_list[data_name] = proc_data

    # write data out
    for data_name in data_list:
        if AREAS_CONSISTENCY_CHECK[data_name] is None:
            continue
        data_list[data_name].to_csv(join(workdir, f"{data_name}.csv"), index=False)




