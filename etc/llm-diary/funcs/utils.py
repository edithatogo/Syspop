from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger

from funcs import LOCATIONS_CFG


def create_logger(logger_path: str | None = None) -> getLogger:
    """
    Creates and configures a logger instance.

    The logger will output messages to the console and optionally to a file.
    It's configured to show messages with INFO level and above.

    Args:
        logger_path (str | None, optional): If provided, the path to a file
            where logs will also be written. Defaults to None (console only).

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    formatter = Formatter("%(message)s")
    if logger_path is not None:
        console_handler = FileHandler(logger_path)
    else:
        console_handler = StreamHandler(logger_path)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def check_locations(locations_to_check: list[str]) -> bool:
    """
    Checks if all locations in a given list are valid keys in LOCATIONS_CFG.

    Args:
        locations_to_check (list[str]): A list of location names to validate.

    Returns:
        bool: True if all locations are valid, False otherwise.
    """
    return set(locations_to_check).issubset(set(LOCATIONS_CFG.keys()))
