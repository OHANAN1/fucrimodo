import logging

def setup_run_logger(
    log_file_path: str, log_level: int = logging.INFO
) -> logging.Logger:
    # Create a logger
    logger = logging.getLogger('run_logger')
    logger.setLevel(logging.DEBUG)

    # Create a file for the log output
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(log_level)

    # Define how the log messages should look like
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger
