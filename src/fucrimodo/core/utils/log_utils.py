import logging


def clear_existing_handlers(logger: logging.Logger):
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def setup_stage_logger(
    log_file_path: str,
    run_name: str,
    stage_name: str,
    log_level: int = logging.INFO,
) -> tuple[logging.Logger, str]:
    """Set up a logger for each stage of the run.

    :param log_file_path: Path to the log file. Normally the directory of the
            stage.
    :param run_name: Name of the run.
    :param stage_name: Name of the stage.
    :param log_level: Level of the log messages. Default is logging.INFO.

    :return: A tuple containing the logger and the name of the logger.
        The name of the logger is the run name and the stage name separated by
        a underscore {run_name}_{stage_name}_logger.
    """
    # Create logger name
    logger_name = f"{run_name}_{stage_name}_logger"

    # Create a logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    # Clear all existing handlers to avoid duplicates
    clear_existing_handlers(logger)

    # Create a file for the log output
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(log_level)

    # Define how the log messages should look like
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger, logger_name


def setup_run_logger(
    log_file_path: str,
    run_name: str,
    log_level: int = logging.INFO,
    verbose: bool = True,
) -> logging.Logger:
    """Set up a logger for the run.

    :param log_file_path: Path to the log file. Normally the directory of the
            run.
    :param run_name: Name of the run.
    :param log_level: Level of the log messages.
    :param verbose: If True, log messages will be printed to the console
        through the StreamHandler.

    :return: A tuple containing the logger and the name of the logger.
        The name of the logger is the run name separated by an underscore
        {run_name}_logger.
    """
    # Create logger name
    logger_name = f"{run_name}_logger"

    # Create a logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    # Clear all existing handlers to avoid duplicates
    clear_existing_handlers(logger)

    # Create a file for the log output
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(log_level)

    # Define how the log messages should look like
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    # Add a StreamHandler if verbose is True to print to console
    if verbose:
        stream_handler = logging.StreamHandler()
        logger.addHandler(stream_handler)

    return logger
