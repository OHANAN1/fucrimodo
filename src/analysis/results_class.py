from icecream import ic
import os
from src.fucrimodo.utils import ase_database_tools as db_tools
import json
import warnings
import numpy as np
from ase.db.core import Database
import ase
from ase.visualize import view
from tabulate import tabulate


class StageResults():

    def __init__(
        self, run_dir, file_name,
        crystals_db: Database | None = None,
        db_stage_key: str = "stage_id"
    ) -> None:
        """
        Initializes a StageResults object from a stage file.
        If crystals_db is not None, the crystals will be loaded from the db.
        Looks for the stage_id at the key db_stage_key to filter the crystals.
        """

        self.stage_file_path = os.path.join(run_dir, file_name)
        if not os.path.exists(self.stage_file_path):
            raise FileNotFoundError(
                f"File {file_name} does not exist in {run_dir}"
            )

        self.name = file_name[:-len(".json")]
        self.id = int(self.name.split("_")[-1])

        self.stage_dict = self.__load_from_file(self.stage_file_path)

        if crystals_db is not None:
            db_data = self.__get_crys_and_key_vals_from_db(
                crystals_db, db_stage_key
            )
            self.crystals, self.key_value_pairs = db_data

    def __get_crys_and_key_vals_from_db(
        self, crystals_db: Database, db_stage_key: str
    ) -> tuple[list, list]:
        """
        Returns the crystals and key value pairs from the database for the
        stage.
        """
        crystals = []
        key_value_pairs = []

        def filter_stage(row):
            if hasattr(row, "key_value_pairs"):
                if db_stage_key in row.key_value_pairs.keys():
                    return row.key_value_pairs[db_stage_key] == self.id
                else:
                    return False
            elif hasattr(row, "is_target"):
                return False

            else:
                warnings.warn(
                    f"Could not find key {db_stage_key} in row {row}."
                )

        for row in crystals_db.select(filter=filter_stage):
            crystals.append(row.toatoms())
            key_value_pairs.append(row.key_value_pairs)


        if len(crystals) == 0:
            raise ValueError(
                f"Could not find any crystals for stage {self.name}."
            )

        if len(key_value_pairs) == 0:
            raise ValueError(
                f"Could not find any key value pairs for stage {self.name}."
            )

        return crystals, key_value_pairs

    def __load_from_file(
        self, stage_file_path: str
    ) -> dict[str, dict[str, list]]:
        with open(stage_file_path, "r") as f:
            stage_dict = json.load(f)

        return stage_dict



class RunResults(): 
    """
    Class to handle the results of a run.
    Also loads target, it assumes that it is at id 1 in the crystals database.
    Uses base name of run_dir as run_name if run_name is not given.
    """
    def __init__(
        self, 
        run_dir: str,
        run_name: str | None = None
    ) -> None:
        self.run_dir = run_dir
        self.stages: list[StageResults] = []

        if run_name is not None:
            self.run_name = run_name
        else:
            self.run_name = os.path.basename(run_dir)

        data = self.__load_from_files(run_dir)
        self.crystals_db, self.run_info, self.stages = data
        self.n_stages = len(self.stages)

        self.target_crystal = self.crystals_db.get(id=1).toatoms()

    def __load_from_files(
        self, run_dir: str
    ) -> tuple[Database, dict | None , list]:
        file_names = os.listdir(run_dir)

        if "crystals.db" not in file_names:
            raise FileNotFoundError(
                f"Could not find crystals.db in {run_dir}."
            )
        else:
            crystals_db = db_tools.connect_to_existing_database(
                database_path=os.path.join(run_dir, "crystals.db")
            )

        run_info = None
        if "run_info.json" not in file_names:
            warnings.warn(
                f"Could not find run_info.json in {run_dir}."
            )
        else:
            with open(os.path.join(run_dir, "run_info.json"), 'r') as json_file:
                run_info = json.load(json_file)

        stages = []
        for file_name in file_names:
            if file_name.startswith("stage_") and file_name.endswith(".json"):
                stage = StageResults(run_dir, file_name, crystals_db)
                stages.append(stage)

        if len(stages) == 0:
            warnings.warn(
                f"Could not find any stages in {run_dir}."
            )

        sorted_stages = sorted(stages, key=lambda x: x.id)

        return crystals_db, run_info, sorted_stages


if __name__ == "__main__":

    import sys

    try:
        run_dir = sys.argv[1]
    except IndexError:
        print("Please use as: python path/to/script.py path/to/run_dir")
        sys.exit(1)

    if not os.path.exists(run_dir):
        print("Path does not exist")
        sys.exit(1)

    run = RunResults(run_dir)
    similarity_key = "SimilarityToTargetSOAPFitness_RBFSimilarity"


    # def get_max_value_of_all_stage_statistics(
    #     self, statistics_key: str
    # ) -> tuple[int, StageResults]:
    #     """
    #     Returns the max value of the value_type "max" for a specific statistic.
    #     """
    #     if statistics_key not in self.get_shared_statistics_keys():
    #         raise KeyError(
    #             f"Could not find key {statistics_key} in all stages."
    #             f"Possible keys: {self.get_shared_statistics_keys()}."
    #         )
    #
    #     max_values = []
    #     for stage in self.stages:
    #         max_values.append(
    #             stage.get_max_statistic_value(statistics_key)
    #         )
    #
    #     max_value_index = np.argmax(max_values)
    #     return max_values[max_value_index], self.stages[max_value_index]

    # def get_best_crystals(
    #     self, statistics_key: str
    # ) -> tuple[list[ase.Atoms], list[dict]]:
    #     """
    #     Returns the best crystals and key value pairs for a given key for 
    #     all stages.
    #     Key needs to be in the crystals database.
    #     Values are returned in the order of the stages.
    #     """
    #     best_crystals = []
    #     key_value_pairs = []
    #     for stage in self.stages:
    #         best_crystal, key_value_pair = stage.get_best_crystal(statistics_key)
    #         best_crystals.append(best_crystal)
    #         key_value_pairs.append(key_value_pair)
    #
    #     return best_crystals, key_value_pairs

    # def get_main_statistic_values_of_all_stages(
    #     self, 
    #     value_type: str | None = None, 
    #     skip_first_value: bool = True,
    #     statistics_key: str | None = None,
    # ) -> list[int] | dict[str, list]:
    #     """
    #     Returns the combines values for the main statistic for all stages
    #     with value_type: "max", "min", "avg" or "std".
    #     If value_type = None, a dict with all types as keys and the
    #     corresponding values will be returned.
    #
    #     Set skip_first_value to True to skip the first value of each stage,
    #     since it is only the initial calculation of the statistic.
    #     """
    #
    #     if statistics_key is None and self.main_statistics_key is not None:
    #         statistics_key = self.main_statistics_key
    #     elif statistics_key is None:
    #         raise ValueError(
    #             "No statistics key given and no main statistics key set."
    #         )
    #
    #     if value_type == None:
    #         values_dict = {}
    #         for stage in self.stages:
    #             values = stage.get_statistics_values(
    #                 statistics_key, value_type
    #             )
    #
    #             if not isinstance(values, dict):
    #                 raise ValueError(
    #                     f"Unexpected format of values for {statistics_key}."
    #                 )
    #
    #             for key, value in values.items():
    #
    #                 if skip_first_value:
    #                     value = value[1:]
    #
    #                 if key not in values_dict.keys():
    #                     values_dict[key] = value
    #                 else:
    #                     values_dict[key].extend(value)
    #
    #         return values_dict
    #
    #     else:
    #         values = []
    #         for stage in self.stages:
    #             stage_values = stage.get_statistics_values(
    #                 statistics_key, value_type
    #             )
    #             if not isinstance(stage_values, list):
    #                 raise ValueError(
    #                     f"Unexpected format of values for {statistics_key}."
    #                 )
    #
    #             if skip_first_value:
    #                 stage_values = stage_values[1:]
    #
    #             values.extend(stage_values)
    #
    #         return values
    #
    # def get_max_statistic_value(self, statistics_key: str) -> int:
    #     """
    #     Returns the max value of the value_type "max" for a specific statistic.
    #     """
    #     max_values = self.get_statistics_values(
    #         statistics_key, value_type="max"
    #     )
    #
    #     if isinstance(max_values, list):
    #         return max(max_values)
    #     else:
    #         raise ValueError(
    #             f"Unexpected format of max values for {statistics_key}."
    #         )
    #
    #
    # def calculate_number_of_generations(self) -> int:
    #     """
    #     Returns the number of generations for the statistics in the stage.
    #     Subtracts 1 from the length of the values list, since the first
    #     value is the initial value.
    #     """
    #     n_generations = []
    #     for key, value in self.__stage_dict.items():
    #         if isinstance(value, dict):
    #             for subkey, subvalue in value.items():
    #                 n_generations.append(
    #                     len(subvalue) - 1
    #                 )
    #
    #     if len(n_generations) == 0:
    #         return 0
    #     elif all(n == n_generations[0] for n in n_generations):
    #         return n_generations[0]
    #     else:
    #         raise ValueError(
    #             f"Number of generations is not the same for all statistics."
    #             f"Values: {n_generations}"
    #         )
# def collect_stage_results(folder_path: str) -> dict[str, dict]:
#     """
#     Collects all runs from the subfolders of a folder.
#     Returns a dictionary:
#     key: run_name
#     value: dictionary with the keys
#
#     subkeys: config_0, config_1, ..., crystals, key_value_pairs
#     """
#     ic("Collecting runs from subfolders of " + folder_path)
#
#     runs_dict = {}
#
#     for root, dirs, files in os.walk(folder_path):
#         ic("Collecting runs from " + root)
#         run_name = os.path.basename(root)
#
#         if (
#             run_name == "." or
#             run_name == "" or
#             run_name == "results" or
#             run_name == "stage_plots" or
#             run_name == "best_and_target_crystals"
#         ):
#             continue
#
#         run_dict = {}
#         for file in files:
#             if (
#                 file.startswith(".") or
#                 file.endswith(".py") or
#                 file.endswith(".ipynb") or
#                 file.endswith(".html") or
#                 file.endswith(".log")
#             ):
#                 continue
#
#             elif file.startswith("stage_") and file.endswith(".json"):
#                 try:
#                     with open(os.path.join(root, file), 'r') as json_file:
#                         run_dict[file[:-len(".json")]] = json.load(json_file)
#
#                 except Exception as e:
#                     print(f"Fehler beim Lesen von {file}: {e}")
#
#             elif file == "crystals.db":
#
#                 crystals_db = db_tools.connect_to_existing_database(
#                     database_path=os.path.join(root, file)
#                 )
#                 db_data = db_tools.get_crystals_and_key_value_pairs_from_database(  # noqa
#                     crystals_db
#                 )
#                 run_dict["crystals"] = db_data[0]
#                 run_dict["key_value_pairs"] = db_data[1]
#
#             elif file == "run_info.json":
#                 try:
#                     with open(os.path.join(root, file), 'r') as json_file:
#                         run_dict["run_info"] = json.load(json_file)
#                 except Exception as e:
#                     print(f"Fehler beim Lesen von {file}: {e}")
#
#             else:
#                 warnings.warn(
#                     f"File {file} not recognized. Skipping."
#                 )
#                 continue
#
#         if "crystals" not in run_dict.keys():
#             warnings.warn(
#                 f"Crystals not found in run {run_name}. Skipping."
#             )
#             continue
#         if "key_value_pairs" not in run_dict.keys():
#             warnings.warn(
#                 f"Key value pairs not found in run {run_name}. Skipping."
#             )
#             continue
#         if "stage_1" not in run_dict.keys():
#             warnings.warn(
#                 f"Configurations not found in run {run_name}. Skipping."
#             )
#             continue
#         if "run_info" not in run_dict.keys():
#             warnings.warn(
#                 f"Run info not found in run {run_name}. Skipping."
#             )
#             continue
#         else:
#             runs_dict[run_name] = run_dict
#
#     ic("Done!")
#     ic()
#     return runs_dict
#
