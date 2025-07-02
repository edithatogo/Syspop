from logging import getLogger

from llama_cpp import Llama
from numpy.random import choice as numpy_choice
from pandas import DataFrame, date_range, to_datetime
from pandas import concat as pandas_concat

from funcs import DEFAULT_MODEL_NAME, LOCATIONS_CFG, PROMPT_QUESTION

logger = getLogger()


def prompt_llm(
    agent_features: dict = {
        "age": "18-65",
        "gender": "male",
        "work_status": "employed",
        "income": "middle income",
        "time": "weekend",
        "others": "",
    },
    model_path: str = DEFAULT_MODEL_NAME,
    max_tokens: int = 1500,
    temperature: float = 0.7,
    top_p: float = 0.7,
    top_k: float = 75,
    n_gpu_layers: int = 256,
    gender: str = numpy_choice(["male", "female"]),
    print_log: bool = False,
) -> dict:
    """
    Prompts a Large Language Model (LLM) to generate a 24-hour diary schedule.

    Args:
        agent_features (dict, optional): A dictionary describing the agent's
            characteristics (age, gender, work_status, income, time_period, others, locations).
            Defaults to a predefined worker profile.
        model_path (str, optional): Path to the LlamaCPP model file.
            Defaults to DEFAULT_MODEL_NAME.
        max_tokens (int, optional): Maximum number of tokens for the LLM to generate.
            Defaults to 1500.
        temperature (float, optional): Controls randomness in LLM generation. Higher values
            mean more randomness. Defaults to 0.7.
        top_p (float, optional): Nucleus sampling parameter. Defaults to 0.7.
        top_k (int, optional): Top-k sampling parameter. Defaults to 75.
        n_gpu_layers (int, optional): Number of GPU layers to use for LLM inference.
            Defaults to 256.
        gender (str, optional): Gender of the agent. Defaults to a random choice
            between "male" and "female".
        print_log (bool, optional): If True, prints the LLM's raw output.
            Defaults to False.

    Returns:
        dict: A dictionary representing the generated diary, with keys as table
              headers (e.g., "Hour", "Activity", "Location") and values as lists
              of corresponding entries.
    """

    if agent_features["time"] == "weekend":
        agent_features["work_status"] = "not working or studying today"

    if agent_features["work_status"] == "retired":
        agent_features["work_status"] = ""

    question_fmt = PROMPT_QUESTION.format(
        gender=gender,
        age=agent_features["age"],
        work_status=agent_features["work_status"],
        income=agent_features["income"],
        others=agent_features["others"],
        locations_list=agent_features["locations"],
    ).replace(" ,", "")

    my_llm = Llama(
        model_path=model_path, verbose=False, seed=-1, n_gpu_layers=n_gpu_layers
    )

    texts = my_llm(
        question_fmt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )["choices"]

    s = texts[0]["text"]

    if print_log:
        print(s)

    s = s[s.index("\n\n|") + len("\n\n|") - 1 : s.index("|\n\n") + 1]

    # Split the string into lines
    lines = s.strip().split("\n")

    # Get the headers
    headers = lines[0].split("|")[1:-1]
    headers = [h.strip() for h in headers]

    # Initialize the dictionary
    data = {h: [] for h in headers}

    # Fill the dictionary
    for line in lines[2:]:
        values = line.split("|")[1:-1]
        values = [v.strip() for v in values]
        for h, v in zip(headers, values):
            data[h].append(v)

    return data


def dict2df(data: dict) -> DataFrame:
    """
    Converts a dictionary representation of a diary into a pandas DataFrame,
    ensuring all hours are present and forward-filling missing entries.

    Args:
        data (dict): A dictionary where keys are "Hour", "Activity", "Location"
                     and values are lists of corresponding entries. The "Hour"
                     is expected in "HH:MM" format.

    Returns:
        DataFrame: A DataFrame with a "Time" index (datetime.time objects) and
                   columns "Hour" (integer), "Activity", and "Location".
                   Missing hourly entries are forward-filled. The first hour's
                   location and activity are set to "Home" and "Sleep".
    """

    # Create DataFrame
    df = DataFrame(data)

    # Convert 'Hour' to datetime and set it as index
    df["Hour"] = to_datetime(df["Hour"], format="%H:%M").dt.time
    df.set_index("Hour", inplace=True)

    # Create a new DataFrame with all hours
    all_hours = date_range(start="00:00", end="23:59", freq="H").time
    df_all = DataFrame(index=all_hours)

    # Join the original DataFrame with the new one
    df = df_all.join(df)

    df.iloc[0]["Location"] = "Home"
    df.iloc[0]["Activity"] = "Sleep"
    # Forward fill the missing values
    df.fillna(method="ffill", inplace=True)

    return data_postp(df)


def extract_keyword(value: str) -> str:
    """
    Extracts a recognized location keyword from a string value.

    It splits the input string into words, checks if any word matches a key in
    LOCATIONS_CFG. If multiple keywords are found, one is chosen randomly.
    If no keyword is found, "others" is returned.

    Args:
        value (str): The input string, potentially containing location names.

    Returns:
        str: A recognized location keyword or "others".
    """
    # Split the value into words and filter out the keywords
    words = value.replace(",", "").split()

    keywords_in_value = [word for word in words if word in list(LOCATIONS_CFG.keys())]

    if len(keywords_in_value) == 0:
        keywords_in_value = ["others"]

    # If there are keywords, return a random one; otherwise, return None
    return numpy_choice(keywords_in_value) if keywords_in_value else None


def data_postp(data: DataFrame) -> DataFrame:
    """
    Post-processes the diary DataFrame.

    This involves:
    1. Converting the "Location" column to lowercase.
    2. Extracting recognized location keywords from the "Location" column using `extract_keyword`.
    3. Resetting the index and renaming "index" to "Time".
    4. Extracting the hour from the "Time" column into a new "Hour" column.

    Args:
        data (DataFrame): The input diary DataFrame, typically output from `dict2df`.
                          Expected to have a "Location" column and a "Time" index.

    Returns:
        DataFrame: The post-processed DataFrame with columns "Time", "Hour",
                   "Activity", and "Location".
    """
    data["Location"] = data["Location"].str.lower()
    data["Location"] = data["Location"].apply(extract_keyword)

    data = data.reset_index()
    data = data.rename(columns={"index": "Time"})

    data["Hour"] = data["Time"].apply(lambda x: x.hour)

    return data[["Time", "Hour", "Activity", "Location"]]


def combine_data(data_list: list[DataFrame]) -> DataFrame:
    """
    Combines a list of individual diary DataFrames into a single DataFrame,
    adding a "People_id" column to distinguish between individuals.

    Args:
        data_list (list[DataFrame]): A list of DataFrames, where each DataFrame
                                     represents a single person's diary and is
                                     expected to have a "Time" index.

    Returns:
        DataFrame: A combined DataFrame with columns "Time", "Hour", "Activity",
                   "Location", and "People_id".
    """
    all_data = []
    for people_id in range(len(data_list)):
        proc_data = data_list[people_id]
        proc_data = proc_data.reset_index()
        proc_data = proc_data.rename(columns={"index": "Time"})
        proc_data["People_id"] = people_id
        all_data.append(proc_data)

    all_data = pandas_concat(all_data, ignore_index=True)

    return all_data[["Time", "Hour", "Activity", "Location", "People_id"]]


def update_locations_with_weights(
    data: DataFrame, day_type: str, base_value: str = "home"
) -> DataFrame:
    """
    Adjusts the occurrence of locations in a diary DataFrame based on predefined
    weights in LOCATIONS_CFG.

    For each location type (e.g., "gym", "supermarket") that has a weight defined
    for the given `day_type` in `LOCATIONS_CFG`, this function reduces the
    number of its occurrences. A certain percentage of its occurrences (determined
    by 1.0 - weight) are randomly replaced with `base_value` (typically "home").

    Args:
        data (DataFrame): The input diary DataFrame, expected to have a "Location" column.
        day_type (str): The type of day ("weekday" or "weekend") to determine weights.
        base_value (str, optional): The location to replace other locations with.
                                    Defaults to "home".

    Returns:
        DataFrame: The diary DataFrame with adjusted location occurrences.
    """
    for proc_loc in LOCATIONS_CFG:
        proc_weight = LOCATIONS_CFG[proc_loc]["weight"]

        if proc_weight is None:
            continue

        indices_to_replace = data.index[data["Location"] == proc_loc].tolist()

        if len(indices_to_replace) == 0:
            continue

        num_values_to_replace = int(
            len(indices_to_replace) * (1.0 - proc_weight[day_type])
        )
        indices_to_replace_random = numpy_choice(
            indices_to_replace, size=num_values_to_replace, replace=False
        )

        data.loc[indices_to_replace_random, "Location"] = base_value

    return data


def update_location_name(data: DataFrame) -> DataFrame:
    """
    Updates location names in the diary DataFrame based on a conversion map
    defined in LOCATIONS_CFG.

    For locations that have a "convert_map" in LOCATIONS_CFG (e.g., "office"
    might convert to "company"), this function replaces the original location
    name with a name chosen probabilistically from the conversion map.

    Args:
        data (DataFrame): The input diary DataFrame with a "Location" column.

    Returns:
        DataFrame: The diary DataFrame with updated location names.
    """

    def _replace_with_prob(row, convert_map: dict, target_value: str):
        if row["Location"] == target_value:
            return numpy_choice(
                list(convert_map.keys()),
                p=list(convert_map.values()),
            )
        else:
            return row["Location"]

    all_data_locs = data["Location"].unique()

    for proc_data_loc in all_data_locs:

        if proc_data_loc in list(LOCATIONS_CFG.keys()):
            proc_weight = LOCATIONS_CFG[proc_data_loc]["convert_map"]

            if proc_weight is not None:
                data["Location"] = data.apply(
                    _replace_with_prob,
                    args=(
                        proc_weight,
                        proc_data_loc,
                    ),
                    axis=1,
                )

    return data
