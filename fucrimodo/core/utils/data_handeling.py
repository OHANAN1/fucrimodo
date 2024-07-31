from typing import Callable, Optional, Sequence
import json
from fucrimodo.core.utils.custom_soap import CustomSOAP
from fucrimodo.core.utils import debug_tools
from fucrimodo.core.modules import Mutation, Crossover, PopulationSelection, FitnessFunction, BreakCondition
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

def convert_to_serializable(obj):
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

# ╒══════════════════════════════════════════════════════════╕
#                      Stage Data Class
# ╘══════════════════════════════════════════════════════════╛

class StageData:
    """
    Class for storing and saving all the data about a stage.
    Gets initialized and handled by the RunData class.
    Contains paremeters specific to the stage
    and the hyperparameters for th gridsearch needed to run the stage.
    Stores the results of the structure search.
    """

    def __init__(
        self,
        number_of_generations: int,
        fitness_functions: list[FitnessFunction | tuple[FitnessFunction, float | int]] | list[FitnessFunction] | list[tuple[FitnessFunction, float | int]],  # noqa
        start_population_selection: PopulationSelection,
        mutation_probability: float,
        crossover_probability: float,
        crossover_list: list[Crossover | tuple[Crossover, float]],  # noqa
        mutation_list: Sequence[Mutation | tuple[Mutation, float]],
        break_condition: BreakCondition,
        additional_statistics_func: Callable[[ase.Atoms], float] | None = None,
        add_stats_func_name: str | None = None,
    ) -> None:
        self.start_population_selection = start_population_selection

        self.fitness_functions = []
        self.fitness_weights = ()
        for fit_tuple in fitness_functions:
            if isinstance(fit_tuple, tuple):
                assert len(fit_tuple) == 2
                assert isinstance(fit_tuple[0], FitnessFunction)
                assert isinstance(fit_tuple[1], float | int)
                self.fitness_functions.append(fit_tuple[0])
                self.fitness_weights += (fit_tuple[1],)

            else:
                assert isinstance(fit_tuple, FitnessFunction)
                self.fitness_functions.append(fit_tuple)
                self.fitness_weights += (1.,)

        self.crossover_list = []
        self.crossover_weights = []
        for cross_tuple in crossover_list:
            if isinstance(cross_tuple, tuple):
                assert len(cross_tuple) == 2
                assert isinstance(cross_tuple[0], Crossover)
                assert isinstance(cross_tuple[1], float)
                self.crossover_list.append(cross_tuple[0])
                self.crossover_weights.append(cross_tuple[1])

            else:
                assert isinstance(cross_tuple, Crossover)
                self.crossover_list.append(cross_tuple)
                self.crossover_weights.append(1.)

        self.mutation_list = []
        self.mutation_weights = []
        for mut_tuple in mutation_list:
            if isinstance(mut_tuple, tuple):
                assert len(mut_tuple) == 2
                assert isinstance(mut_tuple[0], Mutation)
                assert isinstance(mut_tuple[1], float)
                self.mutation_list.append(mut_tuple[0])
                self.mutation_weights.append(mut_tuple[1])

            else:
                assert isinstance(mut_tuple, Mutation)
                self.mutation_list.append(mut_tuple)
                self.mutation_weights.append(1.)

        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability
        self.additional_statistics_func = additional_statistics_func
        self.add_stats_func_name = add_stats_func_name
        self.break_condition = break_condition
        self.n_generations = number_of_generations

    def add_run_settings(
        self,
        crystal_database: Database,
        run_dir: str,
        stage_id: int,
        save_n_best_crystals: int,
    ) -> None:
        """
        Adds the run settings to the stage data.
        Gets called by the RunData class automatically.
        """
        self.run_dir = run_dir
        self.stage_id = stage_id
        self.crystal_database = crystal_database
        self.save_n_best_crystals = save_n_best_crystals

    def get_params_dict(self) -> dict:
        """
        Returns the parameters of the stage as a dictionary.
        """
        params_dict = {
            "number of generations": self.n_generations,
            "start pop generation": self.start_population_selection,
            "fitness functions": self.fitness_functions,
            "fitness weights": self.fitness_weights,
            "crossover list": self.crossover_list,
            "crossover weights": self.crossover_weights,
            "crossover probability": self.crossover_probability,
            "mutation list": self.mutation_list,
            "mutation weights": self.mutation_weights,
            "mutation probability": self.mutation_probability,
            "break condition": self.break_condition,
        }
        return params_dict

    def save_crystals_in_db(
        self,
        crystals: list[ase.Atoms],
        key_value_pairs_list: list[dict],
    ) -> None:
        """
        Saves the most similar crystals of the stage in the crystal
        database of the run.
        The tuple contains the crystal and the key value pairs of the crystal.
        Also adds the stage id to the key value pairs.
        """
        i = 0
        for crystal, key_value_pairs_dict in zip(
            crystals, key_value_pairs_list
        ):
            key_value_pairs_dict["stage_id"] = self.stage_id
            self.crystal_database.write(
                crystal,
                key_value_pairs_dict
            )
            i += 1

    def save_log(
        self,
        log: tools.Logbook,
        log_capter_keys: list[str],
    ) -> None:
        self.save_file_path = f"{self.run_dir}/stage_{self.stage_id}.json"

        log_dict = {}
        for key in log_capter_keys:
            log_dict[key] = {}
            log_dict[key]["min"] = log.chapters[key].select("min")
            log_dict[key]["max"] = log.chapters[key].select("max")
            log_dict[key]["avg"] = log.chapters[key].select("avg")
            log_dict[key]["std"] = log.chapters[key].select("std")

        with open(self.save_file_path, "w") as f:
            json.dump(log_dict, f, indent=4, default=convert_to_serializable)


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
        run_dir_name: str|None = None,
        save_n_best_crystals: int = 10,
        soap_params: dict = {
            "species": [],
            "r_cut": 15,
            "n_max": 8,
            "l_max": 8,
            "sigma": 0.5,
        },
        log_enable: bool = True,
    ) -> None:

        self.save_n_best_crystals = save_n_best_crystals

        # Assign fixed soap parameters
        soap_params["periodic"] = True
        soap_params["sparse"] = False
        soap_params["average"] = "inner"
        self.soap_params = soap_params
        self.soap_object = CustomSOAP(
            species=soap_params["species"],
            r_cut=soap_params["r_cut"],
            n_max=soap_params["n_max"],
            l_max=soap_params["l_max"],
            sigma=soap_params["sigma"],
            periodic=soap_params["periodic"],
        )
        self.stage_data_list = []

        self.run_dir = self.init_run_dir(save_dir, run_dir_name)

        self.crystal_database = self.create_crystal_database(self.run_dir)

        if log_enable:
            log_file_path = f"{self.run_dir}/run_log.log"
            debug_tools.setup_logging(log_file_path)

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
        stage_data_list: list[StageData],
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

    def add_stage_data(self, stage_data: StageData) -> None:
        """
        Adds the stage data to the class.
        """
        self.stage_data_list.append(stage_data)

    def get_stage_data(self, stage_id: int) -> StageData:
        """
        Returns the StageData object of the stage with the given id.
        Adds the necessary run_data to the StageData object.
        """
        if self.stage_data_list == []:
            raise ValueError("No stage data in run data.")

        stage_index = stage_id - 1
        stage_data: StageData = self.stage_data_list[stage_index]

        stage_data.add_run_settings(
            run_dir=self.run_dir,
            stage_id=stage_id,
            crystal_database=self.crystal_database,
            save_n_best_crystals=self.save_n_best_crystals
        )

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

            stage_params = self.get_stage_data(id).get_params_dict()
            for key, value in stage_params.items():
                if isinstance(value, float) or isinstance(value, int):
                    stage_params[key] = value

                elif isinstance(value, list):
                    if len(value) == 0:
                        stage_params[key] = "empty list"
                        continue
                    # if isinstance(value[0], tuple):
                    #     value_dict = {}
                    #     for i, item in enumerate(value):
                    #         if isinstance(item[0], str):
                    #             value_dict[item[0]] = item[1]
                    #         else:
                    #             value_dict[f"item_{id}"] = item
                    #     stage_params[key] = value_dict

                    elif isinstance(
                            value[0], int
                        ) or isinstance(
                            value[0], float
                        ):
                        stage_params[key] = value
                        continue

                    value_dict = {}
                    for item in value:
                        if hasattr(item, "__dict__"):
                            value_dict[item.__class__.__name__] = str(
                                item.__dict__
                            )
                        else:
                            value_dict[item.__class__.__name__] = item

                    stage_params[key] = value_dict

                elif hasattr(value, "__dict__"):
                    stage_params[key] = {
                        value.__class__.__name__: str(value.__dict__)
                    }

            stage_info[f"stage_{id}"] = stage_params

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
