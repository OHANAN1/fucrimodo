from typing import Callable, Optional, Sequence
import json
from fucrimodo.core.utils.custom_soap import CustomSOAP
from fucrimodo.core.utils import debug_tools
from fucrimodo.core.modules import PopulationSelection, FitnessFunction, Stage
from fucrimodo.core.utils.class_parser import convert_class_to_writeable_dict
import datetime
import os
import ase
from ase.db.core import Database
from ase import db
import numpy as np
from deap import tools


# ╒══════════════════════════════════════════════════════════╕
#                   Run Info Json Handeling
# ╘══════════════════════════════════════════════════════════╛

def print_run_info_json(save_path: str):
    """
    Prints the run info json file in a human readable form.

    input:
    save_path:
        path to the run directory,
        run_info name gets added automatically.
    """
    with open(f"{save_path}/run_info.json", "r") as f:
        run_info = json.load(f)
    print(json.dumps(run_info, indent=4))


# ╒══════════════════════════════════════════════════════════╕
#                        Run Data Class
# ╘══════════════════════════════════════════════════════════╛

class RunData:
    """
    Class for storing and saving all the data about a run.
    When initializing the Class a directory is created for the run.
    The directory is named after the initialization time of the class.
    """

    def __init__(
        self,
        save_dir: str,
        soap_object: CustomSOAP,
        run_dir_name: str|None = None,
        save_n_best_crystals: int = 10,
        log_enable: bool = True,
        global_statistics_dict: dict[str, Callable[[ase.Atoms], float]] | None = None,
    ) -> None:

        self.save_n_best_crystals = save_n_best_crystals
        self._global_statistics_dict = global_statistics_dict
        self.global_statistics = global_statistics_dict

        # Assign fixed soap parameters
        self.soap_object = soap_object
        self.soap_params = soap_object.get_init_params()
        self.stage_data_list = []

        self.run_dir = self.init_run_dir(save_dir, run_dir_name)

        self.crystal_database = self.create_crystal_database(self.run_dir)

        if log_enable:
            log_file_path = f"{self.run_dir}/run_log.log"
            debug_tools.setup_logging(log_file_path)

    @property
    def global_statistics(self) -> tools.MultiStatistics | None:
        return self._global_statistics

    @global_statistics.setter
    def global_statistics(
        self, global_stats_dict: dict[str, Callable[[ase.Atoms], float]] | None
    ):
        if global_stats_dict is None:
            self._global_statistics = None
            return

        capter_keys = []
        stats_dict = {}

        for key, func in global_stats_dict.items():
            stats_dict[key] = tools.Statistics(
                key=func
            )
            capter_keys.append(key)

        mstats = tools.MultiStatistics(**stats_dict)
        mstats.register("avg", np.mean)
        mstats.register("max", np.max)
        mstats.register("min", np.min)
        mstats.register("std", np.std)

        self._global_statistics = mstats

    @property
    def global_logbook(self) -> tools.Logbook:
        """A logbook for the global statistics.

        The logbook is used to store the global statistics for all
        generations of all stages.
        In addition to the global statistics, the logbook also stores
        the stage id and the generation number for each entry.
        """
        if not hasattr(self, "_global_log"):
            self._global_log = tools.Logbook()

            global_stats_fields = []
            if self.global_statistics is not None:
                global_stats_fields = self.global_statistics.fields

            self._global_log.header = ['stage_id', 'gen'] + global_stats_fields # type: ignore

        return self._global_log

    def get_time_string(self) -> str:
        now = datetime.datetime.now()
        date_string = now.strftime("%Y_%m_%d_H%H_%M_%S")
        return date_string

    def add_start_time(self) -> None:
        """
        Adds the start time as a attribute to the class.
        The current time is used.
        """
        self.start_time = self.get_time_string()

    def add_end_time(self) -> None:
        """
        Adds the end time as a attribute to the class.
        The current time is used.
        """
        self.end_time = self.get_time_string()

    def add_run_settings(
        self,
        stage_data_list: list[Stage],
        verbose: int = 1
    ) -> None:
        """
        Adds all the run data to the class.
        """
        for stage_data in stage_data_list:
            self.add_stage_data(stage_data)

        self.n_stages = len(self.stage_data_list)

        print("Run settings added.", verbose)

    def create_crystal_database(self, run_dir) -> Database:
        """
        Creates a ase database for the most similar crystals that will be
        saved in each stage
        """
        crystal_database = db.connect(f"{run_dir}/crystals.db")

        crystal_database.metadata = {  # type: ignore
            "title": "Database of most similar crystals.",
            "key_description": {
                "stage_id": (
                    "Stage id", "ID of the Stage in the Run", " "
                ),
            },
            "default_columns": [
                "id", "formula", "stage_id",
            ],
        }
        return crystal_database

    def add_crystal_to_database(
        self,
        crystal: ase.Atoms,
        key_value_pairs: dict,
    ) -> None:
        """
        Adds a crystal to the crystal database.
        """
        self.crystal_database.write(crystal, key_value_pairs)

    def init_run_dir(self, save_dir: str, run_dir_name: str | None) -> str:
        """
        Creates a directory for the run.
        The directory is named after the initialization time of the class.
        """
        print()
        if save_dir[-1] != "/":
            save_dir = f"{save_dir}/"

        if run_dir_name is None:
            run_dir_name = self.get_time_string()

        run_dir = f"{save_dir}{run_dir_name}"

        print("Initializing run directory at:")
        print(run_dir)

        os.mkdir(run_dir)

        return run_dir

    def add_stage_data(self, stage_data: Stage) -> None:
        """
        Adds the stage data to the class.
        """
        self.stage_data_list.append(stage_data)

    def get_stage_data(self, stage_id: int) -> Stage:
        """
        Returns the StageData object of the stage with the given id.
        Adds the necessary run_data to the StageData object.
        """
        if self.stage_data_list == []:
            raise ValueError("No stage data in run data.")

        stage_index = stage_id - 1
        stage_data: Stage = self.stage_data_list[stage_index]

        # stage_data.add_run_settings(
        #     run_dir=self.run_dir,
        #     stage_id=stage_id,
        #     crystal_database=self.crystal_database,
        #     save_n_best_crystals=self.save_n_best_crystals
        # )

        return stage_data

    def save_run_info_json(self, verbose: int = 1) -> dict:
        """
        Generates a dictionary containing all the information about the run.
        Dict is then saved as a json file in the save_dir.
        Note: All the parameter classes need a __repr__ method!
        """
        if verbose > 0:
            print("Saving run info json.")

        if not hasattr(self, "start_time"):
            self.start_time = "unknown"
        if not hasattr(self, "end_time"):
            self.end_time = "unknown"

        run_info = {
            "start time": self.start_time,
            "end time": self.end_time,
            "soap parameters": self.soap_params,
        }

        stage_info = {}
        for i in range(self.n_stages):
            id = i + 1

            stage_params = {} # self.get_stage_data(id).get_params_dict()
            stage_info[f"stage_{id}"] = convert_class_to_writeable_dict(
                stage_params
            )

        run_info["stage_info"] = stage_info

        try:
            with open(f"{self.run_dir}/run_info.json", "w") as f:
                json.dump(run_info, f, indent=4)
        except Exception as e:
            print("Error saving run info json.")
            print(e)
            print()
            print("Printing run info, so it can be safed manually:...")
            import time
            time.sleep(4)
            print(run_info)

        if verbose > 2:
            print("Run info saved:")
            print_run_info_json(self.run_dir)
            print()

        return run_info
