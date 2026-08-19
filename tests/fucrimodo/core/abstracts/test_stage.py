import logging
import os
from time import sleep
from typing import Callable

import pytest
from ase.db.core import Database
from deap import tools

from fucrimodo.core import Individual, Population
from fucrimodo.core.abstracts import Stage


class MockStage(Stage):
    def run(
        self,
        population: Population,
        global_log: tools.Logbook,
        global_stats: tools.MultiStatistics | None,
    ) -> Population:
        return population

    def save_results(
        self,
        save_dir: str,
        structures_db: Database,
        global_statistics_dict: dict[str, Callable[[Individual], float]] | None = None,
    ) -> None:
        return None

    @property
    def info_dict(self) -> dict:
        return {}


@pytest.fixture
def mock_stage():
    return MockStage("Friendly name", "Description")


def test_subclass_without_abstracts_cannot_instantiate():
    # does not implement any abstract
    class Incomplete(Stage):
        pass

    with pytest.raises(TypeError):
        Incomplete("name", "description")  # type: ignore


def test_name_and_description(mock_stage):
    assert mock_stage.name == "Friendly name"
    assert mock_stage.description == "Description"


def test_type_method(mock_stage):
    assert mock_stage.type() == "MockStage"


def test_time_tracking(mock_stage):
    # Test that no initial time is set
    with pytest.raises(AttributeError):
        mock_stage.start_time

    mock_stage.set_start_time()

    # End time is start time when not set
    assert mock_stage.start_time == mock_stage.end_time

    # Wait shortly to set end time
    sleep(0.0001)
    mock_stage.set_end_time()

    assert mock_stage.start_time < mock_stage.end_time


def test_logger(mock_stage: Stage, logger: logging.Logger):
    with pytest.raises(AttributeError):
        mock_stage.logger

    mock_stage.logger = logger

    # Check if logger is same object
    assert mock_stage.logger == logger

    # Check if log level can be changed externally
    logger.setLevel(logging.DEBUG)
    assert mock_stage.logger.level == logging.DEBUG


def test_stage_dir(mock_stage: Stage, temp_dir):
    with pytest.raises(AttributeError):
        mock_stage.stage_dir

    # Test that stage dir is automatically created if not set
    try:
        mock_stage.stage_dir = temp_dir
        assert os.path.isdir(temp_dir)
    finally:
        os.rmdir(temp_dir)


def test_id(mock_stage: Stage):
    with pytest.raises(AttributeError):
        mock_stage.id

    mock_stage.id = 1
    assert mock_stage.id == 1
