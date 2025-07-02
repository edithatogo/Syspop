from collections import Counter as collections_counter
from collections import Counter as collections_counter
from copy import deepcopy
from datetime import datetime, timedelta
from logging import getLogger
from os.path import join
from typing import Optional

import numpy
from numpy import array as numpy_array
from numpy.random import choice as numpy_choice
from numpy.random import normal as numpy_normal
from pandas import DataFrame
from pandas import merge as pandas_merge
from pandas import read_parquet as pandas_read_parquet

from syspop.python import DIARY_CFG, MAPING_DIARY_CFG_LLM_DIARY
from syspop.python.utils import merge_syspop_data, round_a_datetime

logger = getLogger()


def _get_updated_weight(target_value: int, target_weight: dict):
    """Get updated weight from age_weight and time_weight

    Args:
        target_value (int): For example, age like 13
        target_weight (dict): age update weight such as:
            {0-3: 123, 23-123: 123, ...}
    Returns:
        _type_: _description_
    """
    if target_weight is None:
        return 1.0

    for key in target_weight:
        start_target_weight, end_target_weight = map(int, key.split("-"))
        if start_target_weight <= target_value <= end_target_weight:
            return target_weight[key]
    return 1.0


def create_diary_single_person(
    ref_time: datetime = datetime(1970, 1, 1, 0),
    time_var: Optional[numpy.ndarray] = numpy_normal(0.0, 1.5, 100),
    activities: dict = DIARY_CFG["default"],
) -> dict:
    """
    Generates a 24-hour diary for a single person based on activity probabilities
    and time constraints defined in the `activities` configuration.

    The diary is generated hour by hour. For each hour, it identifies available
    activities based on their 'time_ranges'. An activity is then chosen
    probabilistically based on its 'weight' and 'time_weight' for the current hour.
    If the sum of probabilities for available activities is less than 1.0,
    activities from 'random_seeds' are considered to fill the remaining probability.

    Args:
        ref_time (datetime, optional): The reference start datetime for the diary.
            Only the hour component is used for time calculations.
            Defaults to datetime(1970, 1, 1, 0).
        time_var (numpy_array | None, optional): An array of values used to
            randomize the start/end times of activity time ranges. If None,
            no randomization is applied. Defaults to a normal distribution.
        activities (dict, optional): A dictionary defining activities, their weights,
            time ranges, time-specific weights, and max occurrences. Expected
            structure is similar to `DIARY_CFG["default"]`.

    Returns:
        dict: A dictionary where keys are integer hours (0-23) and values are
              the names of the selected activities for that hour.
    """
    ref_time_start = ref_time
    ref_time_end = ref_time + timedelta(hours=24)

    output = {}
    ref_time_proc = ref_time_start
    while ref_time_proc < ref_time_end:
        # Get all activities that can be chosen at this time
        available_activities = []
        for activity in activities:

            if activity == "random_seeds":
                continue

            for start, end in activities[activity]["time_ranges"]:
                time_choice = abs(numpy_choice(time_var)) if time_var is not None else 0
                start2 = round_a_datetime(
                    ref_time + timedelta(hours=start - time_choice)
                )
                end2 = round_a_datetime(ref_time + timedelta(hours=end + time_choice))

                if start2 <= ref_time_proc < end2:

                    if activities[activity]["max_occurrence"] is None:
                        available_activities.append(activity)
                    else:
                        activity_counts = dict(collections_counter(output.values()))
                        if activity not in activity_counts:
                            available_activities.append(activity)
                        else:
                            if (
                                activity_counts[activity]
                                <= activities[activity]["max_occurrence"]
                            ):
                                available_activities.append(activity)

        if available_activities:
            # Choose an activity based on the probabilities
            available_probabilities = [
                activities[proc_activity]["weight"]
                * _get_updated_weight(
                    ref_time_proc.hour, activities[proc_activity]["time_weight"]
                )
                for proc_activity in available_activities
            ]

            total_p = sum(available_probabilities)

            if total_p < 1.0:
                available_activities.extend(activities["random_seeds"])
                remained_p = 1.0 - total_p
                remained_p = remained_p / (len(activities["random_seeds"]))
                remained_p = len(activities["random_seeds"]) * [remained_p]
                available_probabilities.extend(remained_p)
                total_p = sum(available_probabilities)

            # scale up the probability to 1.0
            available_probabilities = numpy_array(available_probabilities)
            available_probabilities /= total_p

            activity = numpy_choice(available_activities, p=available_probabilities)

            # Add the activity to the diary
            output[ref_time_proc.hour] = activity

        else:
            try:
                activity_list = list(activities.keys())
                activity_list.remove("random_seeds")
            except ValueError:
                pass
            output[ref_time_proc.hour] = numpy_choice(activity_list)

        ref_time_proc += timedelta(hours=1)

    return output


def update_weight_by_age(activities_input: dict, age: int) -> dict:
    """
    Adjusts the 'weight' of activities in a configuration dictionary based on
    the provided age and age-specific weight multipliers.

    For each activity in `activities_input` (except "random_seeds"), this function
    multiplies its base 'weight' by an age-specific factor obtained from
    `_get_updated_weight` using the activity's 'age_weight' map.

    Args:
        activities_input (dict): The input activity configuration dictionary.
            Each activity is expected to have a 'weight' and an 'age_weight'
            map (e.g., {'0-5': 0.5, '6-17': 1.0}).
        age (int): The age of the person for whom weights are being updated.

    Returns:
        dict: A deep copy of `activities_input` with adjusted 'weight' values.
    """
    activities_output = deepcopy(activities_input)

    for proc_activity_name in activities_output:
        if proc_activity_name == "random_seeds":
            continue
        activities_output[proc_activity_name]["weight"] *= _get_updated_weight(
            age, activities_output[proc_activity_name]["age_weight"]
        )
    return activities_output


def create_diary(
    syspop_data: DataFrame,
    ncpu: int,
    print_log: bool,
    activities_cfg: dict or None = None,
    llm_diary_data: dict | None = None,
    use_llm_percentage_flag: bool = False,
) -> DataFrame:
    """
    Generates diaries for a synthetic population.

    For each person in `syspop_data`, this function generates a 24-hour diary.
    Diaries can be generated either using a rule-based approach (via
    `create_diary_single_person` with `activities_cfg`) or from pre-generated
    LLM diary data (via `create_diary_single_person_llm` if `llm_diary_data`
    is provided).

    Args:
        syspop_data (DataFrame): DataFrame of the synthetic population. Expected
            columns include 'id', 'age', 'employer' (or 'company'), and 'school'.
        ncpu (int): Number of CPUs (used for logging progress, not for parallelism here).
        print_log (bool): If True, logs progress messages.
        activities_cfg (dict | None, optional): Configuration for rule-based diary
            generation (see `create_diary_single_person`). Defaults to DIARY_CFG.
        llm_diary_data (dict | None, optional): Pre-generated LLM diary data. If provided,
            this method is used for diary generation instead of the rule-based approach.
            Defaults to None.
        use_llm_percentage_flag (bool, optional): If True and `llm_diary_data` is used,
            samples locations based on percentages. Otherwise, samples a full diary.
            Defaults to False.

    Returns:
        DataFrame: A DataFrame containing diaries for all individuals. Columns include
                   'id' and integer hours (0-23) as columns, with activity names
                   as values.
    """

    if activities_cfg is None:
        activities_cfg = DIARY_CFG

    all_diaries = {proc_hour: [] for proc_hour in range(24)}
    all_diaries["id"] = []
    total_people = len(syspop_data)

    for i in range(total_people):

        proc_people = syspop_data.iloc[i]
        if print_log:
            if i % 5000 == 0:
                logger.info(
                    f"Processing [{i}/{total_people}]x{ncpu}: {100.0 * round(i/total_people, 2)}x{ncpu} %"
                )

        if llm_diary_data is None:
            proc_activities = activities_cfg.get(
                "worker"
                if isinstance(proc_people["company"], str)
                else "student" if isinstance(proc_people["school"], str) else "default"
            )

            proc_activities_updated = update_weight_by_age(
                proc_activities, proc_people.age
            )

            output = create_diary_single_person(activities=proc_activities_updated)
        else:
            output = create_diary_single_person_llm(
                llm_diary_data,
                proc_people.age,
                proc_people.employer,
                proc_people.school,
                use_llm_percentage_flag,
            )

        all_diaries["id"].append(proc_people.id)

        for j in output:
            all_diaries[j].append(output[j])

    all_diaries = DataFrame.from_dict(all_diaries)

    return all_diaries


def create_diary_single_person_llm(
    llm_diary_data: dict,
    people_age: int,
    people_company: str,
    people_school: str | None,
    use_percentage_flag: bool,
) -> dict:
    """
    Generates a single person's 24-hour diary using pre-generated LLM diary data.

    Selects an appropriate diary template from `llm_diary_data` based on the
    person's age and work/school status.
    If `use_percentage_flag` is True, it samples locations for each hour based
    on the percentage distribution in the LLM data. Otherwise, it samples a
    complete diary from one of the LLM-generated examples for that person type.
    Finally, it maps LLM-specific location names to standard Syspop location
    names using `MAPING_DIARY_CFG_LLM_DIARY`.

    Args:
        llm_diary_data (dict): A dictionary containing pre-generated LLM diaries.
            Expected to have keys like "percentage" (for probabilistic sampling)
            and "data" (for sampling full diaries), each containing sub-dictionaries
            for different person types (e.g., "toddler", "worker", "student").
        people_age (int): The age of the person.
        people_company (str | None): The company of the person, if employed.
        people_school (str | None): The school of the person, if a student.
        use_percentage_flag (bool): If True, sample locations hour by hour based
            on percentages. If False, sample a full existing diary.

    Returns:
        dict: A dictionary where keys are integer hours (0-23) and values are
              the names of the selected (and mapped) locations for that hour.
    """

    if use_percentage_flag:
        selected_llm_diary_data = llm_diary_data["percentage"]
    else:
        selected_llm_diary_data = llm_diary_data["data"]

    if people_age < 6:
        proc_llm_data = selected_llm_diary_data["toddler"]
    elif people_age > 65:
        proc_llm_data = selected_llm_diary_data["retiree"]
    elif people_company is not None:
        proc_llm_data = selected_llm_diary_data["worker"]
    elif people_school is not None:
        proc_llm_data = selected_llm_diary_data["student"]
    else:
        proc_llm_data = selected_llm_diary_data["not_in_employment"]

    if use_percentage_flag:
        output = {}
        for hour in proc_llm_data.index:
            probabilities = proc_llm_data.loc[hour]
            location = numpy_choice(probabilities.index, p=probabilities.values)
            output[hour] = location
    else:
        selected_people_id = numpy_choice(list(proc_llm_data["People_id"].unique()))
        selected_data = proc_llm_data[proc_llm_data["People_id"] == selected_people_id]

        selected_data["group"] = (
            (selected_data["Hour"].diff() != 1).astype(int).cumsum()
        )

        selected_people_group_id = numpy_choice(list(selected_data["group"].unique()))

        selected_data = selected_data[
            selected_data["group"] == selected_people_group_id
        ]

        output = dict(
            zip(selected_data["Hour"].tolist(), selected_data["Location"].tolist())
        )

    """
    all_unique_locs = []
    for proc_key_loc in list(DIARY_CFG.keys()):
        if proc_key_loc == "random_seeds":
            continue
        all_unique_locs.extend(list(DIARY_CFG[proc_key_loc].keys()))

    all_unique_locs = list(set(all_unique_locs))
    """
    # Initialize an empty dictionary to store the converted values
    updated_output = {}

    # Iterate through the items in dict B
    for key, value in output.items():
        # Iterate through the items in dict A to find the key
        for a_key, a_value in MAPING_DIARY_CFG_LLM_DIARY.items():
            if value in a_value:
                # Assign the key from dict A to the converted dictionary
                updated_output[key] = a_key
                break
            updated_output[key] = value

    return updated_output


def quality_check_diary(
    synpop_data: DataFrame,
    diary_data: DataFrame,
    diary_to_check: list = ["school"],
) -> DataFrame:
    """
    Performs a quality check on generated diaries. If an agent's diary indicates
    they are at a location type (e.g., "school") but the agent does not have
    that specific location assigned in `synpop_data` (e.g., no specific school ID),
    that diary entry is changed to "household".

    Args:
        synpop_data (DataFrame): The main synthetic population DataFrame, indexed by agent ID.
                                 It should contain columns corresponding to location types
                                 listed in `diary_to_check` (e.g., a 'school' column
                                 with specific school IDs or None).
        diary_data (DataFrame): The diary DataFrame, where columns are agent IDs and
                                rows are hours, with location type names as values.
        diary_to_check (list, optional): A list of location types (column names in
                                         `synpop_data`) to validate against.
                                         Defaults to ["school"].

    Returns:
        DataFrame: The `diary_data` DataFrame with corrected diary entries.
    """

    def _check_diary(proc_people_diary: DataFrame, default_place: str = "household"):

        proc_people_id = proc_people_diary["id"]
        proc_people_attr = synpop_data.loc[proc_people_id]
        # Iterate through each hour of the person's diary
        for proc_hr in range(24):
            activity_type = proc_people_diary.iloc[proc_hr]
            # Check if the activity type is one that needs validation (e.g., "school")
            if activity_type in diary_to_check:
                # If the agent is supposed to be at this activity type,
                # but doesn't have a specific instance of it assigned in synpop_data
                # (e.g., at "school" but proc_people_attr["school"] is None),
                # then change the activity to the default_place (e.g., "household").
                if proc_people_attr.get(activity_type) is None:
                    proc_people_diary.at[proc_hr] = default_place
        return proc_people_diary

    return diary_data.apply(_check_diary, axis=1)


def map_loc_to_diary(output_dir: str):
    """
    Maps generic location types in diaries to specific location instances (names/IDs)
    assigned to each agent and saves the final diary.

    This function reads the merged synthetic population data (which includes
    assigned specific locations like a particular school ID or supermarket name for
    each agent) and the diary data (which contains generic location types like
    "school" or "supermarket"). It then replaces the generic types in each agent's
    diary with their assigned specific location instance for each hour.

    The final diary, with specific locations, is saved as "syspop_diaries.parquet".

    Args:
        output_dir (str): The directory containing the synthetic population Parquet
                          files (e.g., "syspop_base.parquet", "syspop_school.parquet")
                          and "syspop_diaries_type.parquet". The output
                          "syspop_diaries.parquet" will also be saved here.

    Raises:
        Exception: If a diary location type for an agent cannot be found as an
                   attribute in the agent's synthetic population data (and is not
                   in `known_missing_locs`).
    """

    # syn_pop_path = join(output_dir, "syspop_base.parquet")
    # synpop_data = pandas_read_parquet(syn_pop_path)

    synpop_data = merge_syspop_data(
        output_dir, ["base", "travel", "shared_space", "household", "work", "school"]
    )
    diary_type_data = pandas_read_parquet(
        join(output_dir, "syspop_diaries_type.parquet")
    )

    time_start = datetime.utcnow()

    def _match_person_diary(
        proc_people: DataFrame,
        known_missing_locs: list = ["gym", "others", "outdoor"],
        proc_diray_mapping: dict = {  # Maps diary types to synpop_data column names
            "kindergarten": "school", # e.g., diary "kindergarten" maps to "school" column
            "company": "employer"     # diary "company" maps to "employer" column
        }
    ):
        proc_people_id = proc_people["id"]
        proc_people_attr = synpop_data.loc[proc_people_id]

        for proc_hr in range(24):
            proc_diray = proc_people.iloc[proc_hr]
            proc_diray = proc_diray_mapping.get(proc_diray, proc_diray)
            #if proc_diray == "travel":
            #    proc_people_attr_value = proc_people_attr["public_transport_trip"]

            try:
                proc_people_attr_value = numpy_choice(
                    proc_people_attr[proc_diray].split(",")
                )
            except (
                KeyError,
                AttributeError,
            ):  # For example, people may in the park from the diary,
                # but it's not the current synthetic pop can support
                if proc_diray in known_missing_locs:
                    proc_people_attr_value = None
                else:
                    raise Exception(
                        f"Not able to find {proc_diray} in the person attribute ..."
                    )
                # proc_people_attr_value = numpy_choice(
                #    proc_people_attr[default_place].split(",")
                # )

            proc_people.at[str(proc_hr)] = proc_people_attr_value

        return proc_people

    diary_data = diary_type_data.apply(_match_person_diary, axis=1)
    time_end = datetime.utcnow()

    logger.info(
        f"Completed within seconds: {(time_end - time_start).total_seconds()} ..."
    )

    diary_data = diary_data.melt(id_vars="id", var_name="hour", value_name="spec")
    diary_type_data = diary_type_data.melt(
        id_vars="id", var_name="hour", value_name="spec"
    )
    diary_data = pandas_merge(
        diary_data, diary_type_data, on=["id", "hour"], how="left"
    )

    diary_data = diary_data.rename(columns={"spec_x": "location", "spec_y": "type"})
    diary_data = diary_data[["id", "hour", "type", "location"]]

    diary_data.to_parquet(join(output_dir, "syspop_diaries.parquet"), index=False)
