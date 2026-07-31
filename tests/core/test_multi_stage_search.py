from numpy.lib.arraysetops import isin
import pytest

import logging
import os
import numpy as np
from fucrimodo.core import MultiStageSearch, multi_stage_search
from fucrimodo.core.modules import Stage
import ase.db
from ase.db.core import Database
from deap import tools
import datetime
import json
import time


def get_multi_stage_search(tmp_path, name=None):  # Uses pytest fixture
    return MultiStageSearch(
        save_dir=tmp_path,
        target_features=np.array([1, 2, 3]),
        descriptor_object=None,
        descriptive_name=name,
        description="This is only for testing",
        global_statistics_dict=None,
        log_level=logging.INFO,
        verbose=False,
    )


def test_name(tmp_path):
    # Test that name is set automatically if not provided
    multi_stage_search = get_multi_stage_search(tmp_path, name=None)
    assert multi_stage_search.name is not None

    # Test setting name manually
    multi_stage_search = get_multi_stage_search(tmp_path, name="manual_name")
    assert multi_stage_search.name == "manual_name"


def test_run_dir(tmp_path):
    # Test that run_dir is generated from save dir and run name
    multi_stage_search = get_multi_stage_search(tmp_path, name="run_name")
    assert multi_stage_search.run_dir == os.path.join(tmp_path, "run_name")

    # Check that creating a run in the same location with the same name leads
    # to a file conflict
    with pytest.raises(FileExistsError):
        multi_stage_search = get_multi_stage_search(tmp_path, name="run_name")


def test_logger(tmp_path):
    # Test that logger writes to file in run_dir
    multi_stage_search = get_multi_stage_search(tmp_path, name="run_name")

    # Check log file exists
    logger_path = os.path.join(multi_stage_search.run_dir, "run.log")
    assert os.path.isfile(logger_path)

    # Check if line count changes if I add log entry
    with open(logger_path, "r") as f:
        line_count_old = sum(1 for _ in f)
    multi_stage_search.logger.info("New entry.")
    with open(logger_path, "r") as f:
        line_count_new = sum(1 for _ in f)
    assert line_count_new - line_count_old == 1

    # Check if changing log level affects logger
    assert multi_stage_search.log_level == multi_stage_search.logger.level
    assert multi_stage_search.logger.level == logging.INFO
    multi_stage_search.log_level = logging.WARN
    assert multi_stage_search.logger.level == logging.WARN


def test_structures_db(tmp_path, ind_slab):
    multi_stage_search = get_multi_stage_search(tmp_path, name="run_name")
    structure_db = multi_stage_search.structures_database
    assert isinstance(structure_db, Database)

    # Write structure to db so it initializes
    structure_db.write(ind_slab)
    assert structure_db.count() == 1

    # Test if structure db is at correct path
    db_path = os.path.join(multi_stage_search.run_dir, "structures.db")
    assert os.path.isfile(db_path)

    # Test if external write also adds structures
    ase.db.connect(db_path).write(ind_slab)
    assert ase.db.connect(db_path).count() == 2
    assert structure_db.count() == 2


def test_global_statistics_and_log_initialization(tmp_path):
    multi_stage_search = get_multi_stage_search(tmp_path)

    def mock_stats(ind):
        return 1.0

    # Test if global logbook gets properly initialized
    assert not hasattr(multi_stage_search, "_global_log")
    assert multi_stage_search._global_statistics is None

    multi_stage_search.global_statistics_dict = {"mock_stats": mock_stats}

    assert hasattr(multi_stage_search, "_global_log")
    assert isinstance(multi_stage_search.global_logbook, tools.Logbook)
    assert isinstance(multi_stage_search.global_statistics, tools.MultiStatistics)


def test_stage_history(tmp_path):
    multi_stage_search = get_multi_stage_search(tmp_path)

    # Test initialization
    assert not hasattr(multi_stage_search, "_stage_history")
    stage_hist = multi_stage_search.stage_history
    assert hasattr(multi_stage_search, "_stage_history")
    assert list(stage_hist.keys()) == ["ID", "relative_save_path"]

    # Test addition of entry
    #  Note: here I need to call a private method
    multi_stage_search._MultiStageSearch__update_stage_history(1, "test_path")  # type: ignore
    assert multi_stage_search.stage_history["ID"] == [1]
    assert multi_stage_search.stage_history["relative_save_path"] == ["test_path"]


def test_stage_handelling(tmp_path, ExampleStage):
    multi_stage_search = get_multi_stage_search(tmp_path)
    example_stage = ExampleStage(name="test_stage", description="")

    # Test if first stage is not present
    expected_stage_dir = os.path.join(multi_stage_search.run_dir, "stage_1")
    assert not os.path.isdir(expected_stage_dir)
    stage_info_path = os.path.join(expected_stage_dir, "info.json")
    assert not os.path.isfile(stage_info_path)

    # Test if stage gets properly initialized
    multi_stage_search.current_stage_id = 1
    multi_stage_search._MultiStageSearch__set_up_stage(example_stage)  # type: ignore

    assert example_stage.id == 1
    assert isinstance(example_stage.start_time, datetime.datetime)
    assert example_stage.stage_dir == expected_stage_dir
    assert os.path.isfile(stage_info_path)
    assert isinstance(example_stage.logger, logging.Logger)

    # Test that stage is added to history
    assert multi_stage_search.stage_history["ID"] == [1]
    assert multi_stage_search.stage_history["relative_save_path"] == [
        os.path.basename(example_stage.stage_dir)
    ]

    # Test data storing of stage infos
    def load_stage_info():
        """Helper for dataloading"""
        with open(stage_info_path, "r") as f:
            data = json.load(f)
        return data

    #  There should already be written something
    initial_stage_info = load_stage_info()
    assert initial_stage_info["total_runtime_ms"] == 0

    #  Set end time after 1 ms
    time.sleep(0.001)
    example_stage.set_end_time()

    multi_stage_search._MultiStageSearch__save_stage_info(example_stage)  # type: ignore
    final_stage_info = load_stage_info()
    assert final_stage_info["total_runtime_ms"] >= 1

    #  The rest stays the same
    assert final_stage_info["id"] == initial_stage_info["id"]
    assert final_stage_info["name"] == initial_stage_info["name"]

    # Test logging of the stage
    stage_logger_path = os.path.join(example_stage.stage_dir, "stage.log")
    #  Check if line count changes if I add log entry
    with open(stage_logger_path, "r") as f:
        line_count_old = sum(1 for _ in f)
    example_stage.logger.info("New entry.")
    with open(stage_logger_path, "r") as f:
        line_count_new = sum(1 for _ in f)
    assert line_count_new - line_count_old == 1


def test_save_results():
    pass


def test_save_info():
    pass


def test_run():
    pass
