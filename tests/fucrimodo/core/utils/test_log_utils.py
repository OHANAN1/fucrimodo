import logging
import os

from fucrimodo.core.utils.log_utils import (
    clear_existing_handlers,
    setup_run_logger,
    setup_stage_logger,
)


def test_clear_existing_handlers():
    logger = logging.getLogger("test_clear_existing_handlers_logger")
    stream_handler = logging.StreamHandler()
    assert len(logger.handlers) == 0

    logger.addHandler(stream_handler)
    assert len(logger.handlers) == 1

    clear_existing_handlers(logger)
    assert len(logger.handlers) == 0


def test_setup_stage_logger(tmp_path):
    log_file_path = os.path.join(tmp_path, "test_logger.log")
    logger = setup_stage_logger(
        log_file_path=log_file_path,
        run_name="example_run_name",
        stage_name="example_stage_name",
        log_level=logging.INFO,
    )

    expected_logger_name = "example_run_name_example_stage_name_logger"

    assert logger.name == expected_logger_name
    assert expected_logger_name in logging.Logger.manager.loggerDict

    # Test if logger writes to correct file
    logger.info("Test")
    with open(log_file_path, "r") as f:
        content = f.read()
    assert "Test" in content


def test_setup_run_logger(tmp_path):
    log_file_path = os.path.join(tmp_path, "test_logger.log")
    logger = setup_run_logger(
        log_file_path=log_file_path,
        run_name="example_run_name",
        log_level=logging.INFO,
        verbose=True,
    )

    expected_logger_name = "example_run_name_logger"

    assert logger.name == expected_logger_name
    assert expected_logger_name in logging.Logger.manager.loggerDict

    # Test if logger writes to correct file
    logger.info("Test")
    with open(log_file_path, "r") as f:
        content = f.read()
    assert "Test" in content

    # Check that logger with verbose True has 2 handlers
    assert len(logger.handlers) == 2

    # Check that logger with verbose False only has 1 handler
    log_file_path = os.path.join(tmp_path, "other_test_logger.log")
    logger = setup_run_logger(
        log_file_path=log_file_path,
        run_name="other_example_run_name",
        log_level=logging.INFO,
        verbose=False,
    )

    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.FileHandler)
