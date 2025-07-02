from copy import deepcopy
from datetime import datetime
from os import makedirs
from os.path import exists, join

from pandas import DataFrame, concat
from pandas import concat as pandas_concat
from pandas import cut as pandas_cut

from syspop.python.base_pop import base_pop_wrapper
from syspop.python.commute import assign_agent_to_commute, create_commute_probability
from syspop.python.diary import (
    create_diary,
    map_loc_to_diary,
    quality_check_diary,
)
from syspop.python.household import create_households, place_agent_to_household
from syspop.python.school import create_school
from syspop.python.shared_space import (
    create_shared_data,
    find_nearest_shared_space_from_household,
    place_agent_to_shared_space_based_on_area,
    place_agent_to_shared_space_based_on_distance,
)
from syspop.python.utils import merge_syspop_data, setup_logging
from syspop.python.work import (
    create_employee,
    create_employer,
    create_income,
    place_agent_to_employee,
    place_agent_to_income,
)

logger = setup_logging(workdir="")


def diary(
    output_dir: str,
    n_cpu: int = 1,
    llm_diary_data: dict | None = None,
    activities_cfg: dict | None = None,
):
    """
    Generates agent diaries and saves them to Parquet files.

    This function orchestrates the diary generation process. It first reads
    previously generated synthetic population data (base, work, school).
    Then, it partitions the population data and generates diaries for each
    partition using the `create_diary` function. After generation, it performs
    a quality check and maps generic diary locations to specific instance names.

    Two main Parquet files are saved in `output_dir`:
    - "syspop_diaries_type.parquet": Diaries with generic location types.
    - "syspop_diaries.parquet": Diaries with specific location names/IDs.

    Args:
        output_dir (str): The directory where synthetic population Parquet files
                          are located and where the output diary Parquet files
                          will be saved.
        n_cpu (int, optional): The number of partitions to split the population
                               data into for processing. This is primarily for
                               managing memory and logging progress, not true
                               parallel processing in this implementation.
                               Defaults to 1.
        llm_diary_data (dict | None, optional): Pre-generated LLM diary data to
            use as a base for diary creation. If None, a rule-based approach
            (`activities_cfg`) is used. Defaults to None.
        activities_cfg (dict | None, optional): Configuration for rule-based
            diary generation if `llm_diary_data` is None. Defaults to None,
            which implies `DIARY_CFG` will be used by `create_diary`.
    """

    start_t = datetime.now()

    logger.info("Diary: reading synthetic population")
    syspop_data = merge_syspop_data(output_dir, ["base", "work", "school"])

    syspop_data_partitions = [
        df for _, df in syspop_data.groupby(pandas_cut(syspop_data.index, n_cpu))
    ]

    logger.info("Diary: start processing diary ...")
    outputs = []
    for i, proc_syspop_data in enumerate(syspop_data_partitions):

        outputs.append(
            create_diary(
                proc_syspop_data,
                n_cpu,
                print_log=True,
                activities_cfg=activities_cfg,
                llm_diary_data=llm_diary_data,
            )
        )

    outputs = pandas_concat(outputs, axis=0, ignore_index=True)

    logger.info("Diary: quality check ...")

    outputs = quality_check_diary(syspop_data, outputs)

    end_t = datetime.now()

    processing_mins = round((end_t - start_t).total_seconds() / 60.0, 2)

    outputs.to_parquet(join(output_dir, "syspop_diaries_type.parquet"))

    logger.info("Diary: start mapping location to diary ...")
    map_loc_to_diary(output_dir)

    logger.info(f"Diary: created within {processing_mins} minutes ...")


def create(
    syn_areas,
    output_dir,
    population: dict = None,
    geography: dict = None,
    household: dict = None,
    work: dict = None,
    commute: dict = None,
    education: dict | None = None,
    shared_space: dict | None = None,
):
    """
    Main orchestrator function to generate a synthetic population and associated datasets.

    This function takes various input data configurations (population structure,
    geography, household composition, work details, commute patterns, education
    facilities, and shared spaces) and processes them to create a detailed
    synthetic population. Each agent in the population is assigned attributes like
    age, gender, ethnicity, household, work/school location, travel methods, and
    specific instances of shared spaces they might visit.

    The generated data is saved into multiple Parquet files in the `output_dir`,
    categorized by data type (e.g., "syspop_base.parquet", "syspop_household.parquet").

    Args:
        syn_areas (list | pd.Series): A list or Series of SA2 area IDs for which the
            synthetic population should be generated.
        output_dir (str): The directory where all output Parquet files will be saved.
        population (dict | None, optional): Dictionary containing the 'structure'
            DataFrame for population.
        geography (dict | None, optional): Dictionary containing 'address' and 'location'
            DataFrames.
        household (dict | None, optional): Dictionary containing the 'composition'
            DataFrame for households.
        work (dict | None, optional): Dictionary containing 'employee', 'employer',
            and 'income' DataFrames.
        commute (dict | None, optional): Dictionary containing 'travel_to_work' and
            'travel_to_school' DataFrames.
        education (dict | None, optional): Dictionary containing 'school' and
            'kindergarten' DataFrames.
        shared_space (dict | None, optional): Dictionary where keys are shared space
            types (e.g., "hospital", "supermarket") and values are DataFrames
            containing their locations and attributes.

    Raises:
        OSError: If the `output_dir` cannot be created.
        KeyError: If expected keys are missing from the input data dictionaries.
    """
    if not exists(output_dir):
        makedirs(output_dir)

    logger.info("----------------------------")
    logger.info("Creating base population ...")
    logger.info("----------------------------")
    population_data = base_pop_wrapper(population["structure"], syn_areas)
    all_areas = list(population_data["area"].unique())

    logger.info("----------------------------")
    logger.info("Creating required data ...")
    logger.info("----------------------------")
    logger.info("Creating household data ...")
    household_data = create_households(
        household["composition"], geography["address"], all_areas
    )

    logger.info("Creating work related data ...")
    commute_data_work = create_commute_probability(
        commute["travel_to_work"], all_areas, commute_type="work"
    )
    employee_data = create_employee(
        work["employee"], commute_data_work.area_work.unique()
    )
    employer_data = create_employer(
        work["employer"],
        geography["address"],
        list(commute_data_work.area_work.unique()),
    )
    income_data = create_income(work["income"])

    logger.info("Creating school related data ...")
    commute_data_school = create_commute_probability(
        commute["travel_to_school"], all_areas, commute_type="school"
    )

    school_data = create_school(
        concat([education["school"], education["kindergarten"]])
    )

    logger.info("Creating shared space related data ...")
    shared_space_data = {}
    shared_space_loc = {}
    for proc_shared_space_name in shared_space:
        shared_space_data[proc_shared_space_name] = create_shared_data(
            shared_space[proc_shared_space_name], proc_shared_space_name
        )

        shared_space_loc[proc_shared_space_name] = (
            find_nearest_shared_space_from_household(
                household_data,
                shared_space_data[proc_shared_space_name],
                geography["location"],
                proc_shared_space_name,
            )
        )

    logger.info("----------------------------")
    logger.info("Creating agents ...")
    logger.info("----------------------------")
    updated_agents = []
    updated_household_data = deepcopy(household_data)
    total_people = len(population_data)
    for i, proc_agent in population_data.iterrows():

        if i % 500.0 == 0:
            logger.info(
                f"Completed: {i} / {total_people}: {int(i * 100.0/total_people)}%"
            )

        # ----------------
        # Work
        # ----------------
        proc_agent = assign_agent_to_commute(
            commute_data_work,
            proc_agent,
            commute_type="work",
            include_filters={"age": [(18, 999)]},
        )
        proc_agent = place_agent_to_employee(employee_data, proc_agent)
        proc_agent = place_agent_to_income(income_data, proc_agent)
        proc_agent = place_agent_to_shared_space_based_on_area(
            employer_data,
            proc_agent,
            "work",
            filter_keys=["business_code"],
            name_key="employer",
            shared_space_type_convert={"work": "employer"},
        )

        # ----------------
        # School
        # ----------------
        proc_agent = assign_agent_to_commute(
            commute_data_school,
            proc_agent,
            commute_type="school",
            include_filters={"age": [(0, 17)]},
        )
        proc_agent = place_agent_to_shared_space_based_on_area(
            school_data,
            proc_agent,
            "school",
            filter_keys=["age"],
            name_key="school",
            weight_key="max_students",
        )

        # ----------------
        # Household
        # ----------------
        proc_agent, updated_household_data = place_agent_to_household(
            updated_household_data, proc_agent
        )

        # ----------------
        # Shared space
        # ----------------
        proc_agent = place_agent_to_shared_space_based_on_distance(
            proc_agent, shared_space_loc
        )

        updated_agents.append(proc_agent)

    updated_agents = DataFrame(updated_agents)

    updated_agents["id"] = updated_agents.index

    output_files = {
        "syspop_base": ["area", "age", "gender", "ethnicity"],
        "syspop_household": ["household"],
        "syspop_travel": ["travel_method_work", "travel_method_school"],
        "syspop_work": ["area_work", "business_code", "employer", "income"],
        "syspop_school": ["area_school", "school"],
        "syspop_shared_space": [
            "hospital",
            "supermarket",
            "restaurant",
            "cafe",
            "department_store",
            "wholesale",
            "fast_food",
            "pub",
            "park",
        ],
    }

    for name, cols in output_files.items():
        output_path = join(output_dir, f"{name}.parquet")
        try:
            updated_agents[["id"] + cols].to_parquet(output_path, index=False)
        except KeyError:
            pass

    household_data.to_parquet(join(output_dir, "household_data.parquet"), index=False)
    employer_data.to_parquet(join(output_dir, "employer_data.parquet"), index=False)
    school_data.to_parquet(join(output_dir, "school_data.parquet"), index=False)
    for shared_space_name in shared_space_data:
        shared_space_data[shared_space_name].to_parquet(
            join(output_dir, f"{shared_space_name}_data.parquet"), index=False
        )
