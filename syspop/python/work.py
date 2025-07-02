
from logging import getLogger
from uuid import uuid4

from pandas import DataFrame, Series

logger = getLogger()

def create_income(income_dataset: DataFrame) -> DataFrame:
    """
    Simply returns the input income dataset.

    This function acts as a placeholder or a simple pass-through for the income data.
    It might be used for consistency in a workflow where other data types
    undergo more complex creation processes.

    Args:
        income_dataset (DataFrame): The input DataFrame containing income data.

    Returns:
        DataFrame: The same input `income_dataset`.
    """
    return income_dataset


def create_employee(employee_data: DataFrame, all_areas: list) -> DataFrame:
    """
    Filters employee data for specified work areas and selects relevant columns.

    Args:
        employee_data (DataFrame): Input DataFrame containing employee counts or
                                   percentages. Expected columns: 'area',
                                   'business_code', 'employee'.
        all_areas (list): A list of area IDs (work areas) to filter by.

    Returns:
        DataFrame: A filtered DataFrame with columns ['area_work', 'business_code',
                   'employee']. The 'area' column from the input is renamed
                   to 'area_work'.
    """
    employee_data = employee_data[employee_data["area"].isin(all_areas)]
    employee_data = employee_data.rename(columns={"area": "area_work"})
    return employee_data[["area_work", "business_code", "employee"]]


def create_employer(employer_dataset: DataFrame, address_data: DataFrame, all_areas: list) -> DataFrame:
    """
    Disaggregates employer counts into individual employer records, assigning each
    a unique ID and a specific address within its SA2 area.

    The input `employer_dataset` contains counts of employers per 'area' and
    'business_code'. This function expands these counts, so if an area has
    N employers of a certain type, N individual employer records are created.
    Each created employer is assigned a random address (latitude, longitude)
    from `address_data` that falls within its 'area'.

    Args:
        employer_dataset (DataFrame): Aggregated employer data. Expected columns:
            'area', 'business_code', 'employer' (count of employers).
        address_data (DataFrame): DataFrame of available addresses with 'area',
                                  'latitude', 'longitude' columns.
        all_areas (list): A list of area IDs to process. `employer_dataset`
                          will be filtered by these areas.

    Returns:
        DataFrame: A DataFrame where each row is an individual employer.
                   Columns: 'area_work' (renamed from 'area'), 'business_code',
                   'latitude', 'longitude', 'employer' (unique ID).
    """
    employer_dataset = employer_dataset[employer_dataset["area"].isin(all_areas)]

    employer_datasets = []
    # Loop through each row in the original DataFrame
    for _, row in employer_dataset.iterrows():
        area = row["area"]
        business_code = row["business_code"]
        count = row["employer"]
        proc_address_data_area = address_data[
            address_data["area"] == area]

        # Create individual records for each household
        for _ in range(count):
            proc_address_data = proc_address_data_area.sample(n=1)
            employer_datasets.append({
                "area_work": int(area),
                "business_code": str(business_code),
                "latitude": float(proc_address_data.latitude),
                "longitude": float(proc_address_data.longitude),
                "employer": str(uuid4())[:6]  # Create a 6-digit unique ID
            })

    return DataFrame(employer_datasets)


def place_agent_to_employee(employee_data: DataFrame, agent: Series) -> Series:
    """
    Assigns a business code to an agent based on their work area and the
    distribution of employees across business codes in that area.

    If the agent has no assigned 'area_work', their 'business_code' is set to None.
    Otherwise, it filters `employee_data` for the agent's 'area_work'.
    If there are no employees in that area, 'business_code' is "Unknown".
    Otherwise, a business code is sampled based on the proportion of employees
    in each business code within that work area.

    Args:
        employee_data (DataFrame): DataFrame of employee counts. Expected columns:
            'area_work', 'business_code', 'employee' (count or proportion).
        agent (Series): The agent to assign a business code to. Must have
                        an 'area_work' attribute.

    Returns:
        Series: The updated agent Series with a 'business_code' attribute assigned
                (str or None or "Unknown").
    """
    if agent.area_work is None:
        selected_code = None
    else:
        proc_employee_data = employee_data[
            employee_data["area_work"] == agent.area_work]
        total_employee = proc_employee_data["employee"].sum()
        if total_employee == 0:
            selected_code = "Unknown"
        else:
            proc_employee_weight = proc_employee_data["employee"] / total_employee
            selected_code = proc_employee_data.sample(
                    n=1,
                    weights=proc_employee_weight)["business_code"].values[0]

    agent["business_code"] = selected_code

    return agent


def place_agent_to_income(income_data: DataFrame, agent: Series) -> Series:
    """
    Assigns an income value (as a string) to an agent based on their
    demographic and work characteristics.

    If the agent has no 'area_work', income is set to None.
    Otherwise, it filters `income_data` based on the agent's 'gender',
    'business_code' (can be a comma-separated list in `income_data`),
    'ethnicity', and 'age' (age bands in `income_data`).
    If a unique match is found, that income 'value' is assigned. If no match
    or multiple matches occur, 'income' is set to "Unknown".

    Args:
        income_data (DataFrame): DataFrame containing income data. Expected columns:
            'gender', 'business_code' (can be "code1, code2"), 'ethnicity',
            'age' (formatted as "age1-age2"), and 'value' (the income amount).
        agent (Series): The agent to assign income to. Expected attributes:
            'area_work', 'gender', 'business_code', 'ethnicity', 'age'.

    Returns:
        Series: The updated agent Series with an 'income' attribute (str or None).

    Raises:
        Exception: If more than one income record matches the agent's criteria.
    """
    if agent.area_work is None:
        selected_income = None
    else:
        income_data[["business_code1", "business_code2"]] = income_data["business_code"].str.split(",", expand=True)
        income_data[["age1", "age2"]] = income_data["age"].str.split("-", expand=True)

        for item in ["business_code1", "business_code2"]:
            income_data[item] = income_data[item].str.strip()
        for item in ["age1", "age2"]:
            income_data[item] = income_data[item].astype(int)

        proc_income_data = income_data[
            (income_data["gender"] == agent.gender) &
            ((income_data["business_code1"] == agent.business_code) | (income_data["business_code2"] == agent.business_code)) &
            (income_data["ethnicity"] == agent.ethnicity) &
            ((agent.age >= income_data["age1"]) & (agent.age <= income_data["age2"]) )]

        if len(proc_income_data) > 1:
            raise Exception("Income data decoding error ...")

        if len(proc_income_data) == 0:
            selected_income = "Unknown"
        else:
            selected_income = proc_income_data["value"].values[0]

        agent["income"] = str(selected_income) # we can't have nan, unknown and a numerical value together in a parquet

    return agent
