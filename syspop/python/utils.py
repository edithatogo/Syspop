from datetime import datetime, timedelta
from functools import reduce as functools_reduce
from logging import INFO, Formatter, StreamHandler, basicConfig, getLogger
from os.path import exists, join

from pandas import DataFrame
from pandas import merge as pandas_merge
from pandas import read_parquet as pandas_read_parquet

logger = getLogger()


def setup_logging(
    workdir: str = "/tmp",
    log_type: str = "syspop",
    start_utc: datetime = datetime.utcnow(),
) -> getLogger:
    """
    Sets up a basic logging configuration for the application.

    Configures a logger that outputs to both the console (StreamHandler) and
    optionally to a log file. The log file is named based on `log_type` and
    the `start_utc` date.

    Args:
        workdir (str, optional): The directory where the log file will be created.
            Defaults to "/tmp".
        log_type (str, optional): A prefix for the log file name.
            Defaults to "syspop".
        start_utc (datetime, optional): The UTC datetime used to timestamp the
            log file name. Defaults to `datetime.utcnow()`.

    Returns:
        logging.Logger: The configured root logger instance.
    """
    formatter = Formatter(
        "%(asctime)s - %(name)s.%(lineno)d - %(levelname)s - %(message)s"
    )

    ch = StreamHandler()
    ch.setLevel(INFO)
    ch.setFormatter(formatter)
    logger_path = join(workdir, f"{log_type}.{start_utc.strftime('%Y%m%d')}")
    basicConfig(filename=logger_path),
    logger = getLogger()
    logger.setLevel(INFO)
    logger.addHandler(ch)

    return logger


def round_a_datetime(dt: datetime) -> datetime:
    """
    Rounds a datetime object to the nearest hour.

    If the minute component is 30 or more, it rounds up to the next hour.
    Otherwise, it truncates to the current hour. Seconds and microseconds
    are always set to zero.

    Args:
        dt (datetime): The datetime object to round.

    Returns:
        datetime: The datetime object rounded to the nearest hour.
    """
    # If minutes are 30 or more, round up to the next hour
    if dt.minute >= 30:
        dt += timedelta(hours=1)
    return dt.replace(minute=0, second=0, microsecond=0)


def merge_syspop_data(data_dir: str, data_types: list[str]) -> DataFrame:
    """
    Merges multiple Syspop Parquet datasets based on a common 'id' column.

    For each data type specified in `data_types`, this function looks for a
    Parquet file named "syspop_{data_type}.parquet" in the `data_dir`.
    It reads all found Parquet files into DataFrames and then performs an
    inner merge on the 'id' column.

    Args:
        data_dir (str): The directory where the Syspop Parquet files are located.
        data_types (list[str]): A list of strings representing the types of data
            to merge (e.g., ["base", "travel", "work"]). This corresponds to
            the suffix in the Parquet file names.

    Returns:
        DataFrame: A single DataFrame containing all merged data. If no files
                   are found or only one is found, the behavior might differ
                   (functools.reduce might raise an error or return the single DataFrame).
    """

    proc_data_list = []
    for required_data_type in data_types:
        proc_path = join(data_dir, f"syspop_{required_data_type}.parquet")

        if exists(proc_path):
            proc_data_list.append(pandas_read_parquet(proc_path))

    return functools_reduce(
        lambda left, right: pandas_merge(left, right, on="id", how="inner"),
        proc_data_list,
    )


def select_place_with_contstrain(
    places: DataFrame,
    constrain_name: str,
    constrain_priority: str,
    constrain_options: list,
    constrain_priority_weight: float = 0.85,
    check_constrain_priority: bool = False,
) -> DataFrame:
    """
    Selects a single row (place) from a DataFrame based on weighted sampling,
    giving priority to a specific `constrain_priority` value within a
    `constrain_name` column.

    If `constrain_priority` is not among `constrain_options` and
    `check_constrain_priority` is False, it falls back to uniform random sampling.

    Args:
        places (DataFrame): The DataFrame from which to sample a place.
        constrain_name (str): The name of the column in `places` that contains the
                              categories to be weighted (e.g., 'ethnicity').
        constrain_priority (str): The specific value within `constrain_name` column
                                  that should receive higher weight.
        constrain_options (list): A list of all possible unique values in the
                                  `constrain_name` column of the `places` DataFrame.
        constrain_priority_weight (float, optional): The weight (probability)
            assigned to `constrain_priority`. Must be between 0 and 1.
            Defaults to 0.85.
        check_constrain_priority (bool, optional): If True, an exception will be
            raised if `constrain_priority` is not found in `constrain_options`.
            If False and not found, uniform sampling is used. Defaults to False.

    Returns:
        DataFrame: A single-row DataFrame representing the selected place.

    Raises:
        Exception: If `constrain_name` is not a column in the `places` DataFrame.
        Exception: If `check_constrain_priority` is True and `constrain_priority`
                   is not in `constrain_options`.
    """
    if constrain_name not in places:
        raise Exception(
            f"The constrain name: {constrain_name} is not in the place_data"
        )

    if constrain_priority not in constrain_options:
        if check_constrain_priority:
            raise Exception(
                f"The agent constrain priority {constrain_priority} ({constrain_name}) "
                "is not part of the provided constrain_options"
            )
        else:
            return places.sample(n=1)

    constrain_weights_others = (1.0 - constrain_priority_weight) / (
        len(constrain_options) - 1
    )

    constrain_weights = [constrain_weights_others] * len(constrain_options)

    constrain_weights[constrain_options.index(constrain_priority)] = (
        constrain_priority_weight
    )

    # Create a weight mapping
    constrain_weights_map = dict(zip(constrain_options, constrain_weights))

    # Assign weights to each row based on category
    df_weights = places[constrain_name].map(constrain_weights_map)

    # Sample one row using weights
    return places.sample(n=1, weights=df_weights)
