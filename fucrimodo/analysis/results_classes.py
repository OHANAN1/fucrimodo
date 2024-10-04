import os
from typing import Any

import pickle
import ase
from deap import tools
from fucrimodo.core.utils import ase_database_tools as db_tools
import json
import warnings
import pickle
import pandas as pd


class StageResults():
    """
    Collects and structures the data that was collected during a given
    stage from a run directory.

    :param run_dir: Path to the directory where the run was saved
    :type run_dir: str
    :param id: ID of the stage that should be analyzed
    :type id: int

    :raises FileNotFoundError: One of the expected files was not found at 
        path :data:`run_dir`. 
        Necessary files are: 'stage_[:data:`id`].json' and 'crystals.db'
    :raises ValueError: If no crystals or key value pairs where found in the 
        database at the database key 'stage_id' with value :data:`id`.
    """
    def __init__(
        self, 
        run_dir: str, 
        id: int,
    ) -> None:
        self._name = f"stage_{id}"
        self._id = id
        self._run_dir = run_dir

        # Load the stage data from the stage directory
        self.dir_path = os.path.join(run_dir, f"stage_{id}")
        if not os.path.exists(self.dir_path):
            raise FileNotFoundError(
                f"Directory stage_{id} does not exist in {run_dir}."
                "Was the desired stage even performed?"
            )

        # Load the info dict of the stage
        self._info_dict = self.__load_dict_from_file("info.json")

        # Load the crystals and key value pairs associated with the stage
        # from the database safed in the run dir
        crystals_db_path = os.path.join(self._run_dir, "crystals.db")
        self._crystals, self._key_value_pairs = self.__get_crystal_data_from_db(
            crystals_db_path=crystals_db_path
        )

    @property
    def crystals(self) -> list[ase.Atoms]:
        """A list of all crystals that where saved during the stage."""
        return self._crystals

    @property
    def key_value_pairs(self) -> list[dict[str, Any]]:
        """A list of the key value pairs of all crystals in the stage.

        Has the same lenght as :attr:`StageResults.crystals`.
        """
        return self._key_value_pairs

    @property
    def fitnesses(self) -> pd.DataFrame:
        """The fitness information and statistics that where tracked during the stage.

        A Dataframe with columns `names`, `weights`, `reprs`, `hashes` and
        `results`.
        Each row corresponds to a specific fitness operator.
        The results entries are dataframes with columns `max`, `min`,
        `avg`, `std` and `gen`.
        """
        # Check if the fitnesses where already loaded
        if not hasattr(self, "_fitnesses"):
            # Load the fitnesses dict from the fitnesses.json file
            fit_dict = self.__load_dict_from_file("fitnesses.json")

            # Load each of the results entries in a Dataframe
            for i in range(len(fit_dict["results"])):
                fit_dict["results"][i] = pd.DataFrame(fit_dict["results"][i])

            # Create the fitnesses Dataframe
            self._fitnesses = pd.DataFrame(fit_dict)

        return self._fitnesses

    @property
    def mutations(self) -> pd.DataFrame:
        """The mutation information and statistics that where tracked during the stage.

        A pandas dataframe with keys `names`, `weights`, `reprs`, `hashes` and
        `results`.
        Each row corresponds to a specific mutation operator.
        The results entry is a Dataframe with columns `called`, `failed`,
        `survivor` and `gen`.
        """
        # Check if the mutations where already loaded
        if not hasattr(self, "_mutations"):
            # Get the dict from the mutations.json file
            mut_dict = self.__load_dict_from_file("mutations.json")

            # Load each of the results entries in a Dataframe
            for i in range(len(mut_dict["results"])):
                mut_dict["results"][i] = pd.DataFrame(mut_dict["results"][i])

            # Create the mutations Dataframe
            self._mutations = pd.DataFrame(mut_dict)
        return self._mutations

    @property
    def crossovers(self) -> pd.DataFrame:
        """The crossover information and statistics that where tracked during the stage.

        A pandas Dataframe with keys `names`, `weights`, `reprs`, `hashes` and 
        `results`.
        Each row corresponds to a specific crossover operator.
        The results entry is a Dataframe with columns `called`, `failed`,
        `survivor` and `gen`.
        """
        # Check if the crossovers where already loaded
        if not hasattr(self, "_crossovers"):
            # Load the crossovers dict from the crossovers.json file
            cross_dict = self.__load_dict_from_file("crossovers.json")

            # Load each of the results entries in a Dataframe
            for i in range(len(cross_dict["results"])):
                cross_dict["results"][i] = pd.DataFrame(cross_dict["results"][i])

            # Create the crossovers Dataframe
            self._crossovers = pd.DataFrame(cross_dict)

        return self._crossovers

    @property
    def name(self) -> str:
        """Name of the stage. Currently autogenerated."""
        return self._name

    @property
    def id(self) -> int:
        """ID of the stage in the run"""
        return self._id

    @property
    def n_generations(self) -> int:
        """Number of generations that the stage performed."""
        # Get the results dict of the first fitness entry and return the
        # last entry of the 'gen' key which is the index of the last generation
        # that was performed.
        return self.fitnesses["results"][0]["gen"][-1]

    @property
    def info_dict(self) -> dict[str, Any]:
        """Information about the stage that was saved in the info.json file.

        For all stages the keys are 'type', 'id', 'name' and 'description'.
        The GAStage has additional keys 'break_condition'.
        """
        return self._info_dict

    def __load_dict_from_file(
        self, file_name: str
    ) -> dict[str, list]:
        """Load a dictionary from a json file with name :data:`file_name` from
        the stage directory :attr:`StageResults.dir_path`

        :param file_name: Name of the file that should be loaded.

        :raises AssertionError: If the file does not exist in the stage 
            directory.

        :return: The loaded dictionary.
        """
        file_path = os.path.join(self.dir_path, file_name)
        assert os.path.exists(file_path), \
            f"File {file_name} does not exist in {self.dir_path}."

        with open(file_path, "r") as f:
            stage_dict = json.load(f)
        return stage_dict

    def __get_crystal_data_from_db(
        self, crystals_db_path: str
    ) -> tuple[list[ase.Atoms], list[dict[str, Any]]]:
        """Collects the crystals and key value pairs from the crystals database.

        Th data is located at :data:`crystals_db_path` and must have the key 
        'stage_id' with value :attr:`StageResults.id`.

        :param crystals_db_path: Path to the crystals database. Normally it is
            located in the run directory and is named 'crystals.db'.

        :raises ValueError: If no crystals or key value pairs where found in the
            database at the database key 'stage_id' with value 
            :attr:`StageResults.id`.

        :return: A tuple with the crystals and key value pairs dictionaries.
        """
        assert os.path.exists(crystals_db_path), \
            f"File crystals.db does not exist in {self._run_dir}."
        crystals_db = db_tools.connect_to_existing_database(crystals_db_path)
        db_data = db_tools.get_data_with_specific_key_value_from_db(
            crystals_db=crystals_db, key="stage_id", value=self.id
        )
        return db_data


class RunResults(): 
    """
    Collects and structures the data that was collected during a run 
    that was saved at :data:`run_dir`.
    Automatically looks for all stages that where performed and loads them in 
    a list of :class:`StageResults` classes.

    :param run_dir: Path to the directory where the run was saved
    :type run_dir: str
    :param run_name: If None uses the base name of the :data:`run_dir`.
    :type run_name: str | None

    :raises FileNotFoundError: The expected files 'crystals.db' was not found 
        at path :data:`run_dir`. 
    """
    def __init__(
        self, 
        run_dir: str,
        run_name: str | None = None
    ) -> None:
        crystal_db_file_path = os.path.join(run_dir, "crystals.db")
        if not os.path.exists(crystal_db_file_path):
            raise FileNotFoundError(
                f"Could not find crystals.db in {run_dir}."
            )

        self.run_dir = run_dir
        self.run_name = run_name

        data = self.__load_from_files(run_dir)
        self._stages, self._run_info = data
        self.n_stages = len(self.stages)

        with open(os.path.join(run_dir, "global_logbook.pickle"), "rb") as f:
            self._global_statistics_log = pickle.load(f)

    @property
    def run_name(self) -> str:
        """
        Name of the run. If set to None, will automatically use the base 
        name of the run directory.
        """
        return self._run_name

    @run_name.setter
    def run_name(self, value: str | None):
        if value == None:
            self._run_name = os.path.basename(self.run_dir)
        else:
            self._run_name = value

    @property
    def global_statistics_log(self) -> tools.Logbook:
        """The global statistics were tracked for each generation during the run.

        Keys are the names of the statistics and the values are dicts
        with the different value types that where tracked for each generation.
        Normally the value types are 'mean', 'std', 'min', 'max'.
        """
        return self._global_statistics_log

    @property
    def run_info(self) -> dict | None:
        """
        Holds the data that is saved in run_info. 
        That is mainly configuration data.
        If run did not finish will return None.
        
        TODO: Make this part of the stages, but first ensure that the run_info
        data is saved in the beginning of a run and not only after it has 
        finished.
        """
        warnings.warn(
            "This should be part of the stages. Read doc-String for more info!"
        )
        return self._run_info

    @property
    def stages(self) -> list[StageResults]:
        """
        Ordered list of the StageResults of stages that could be found 
        in :data:`run_dir`.
        """
        return self._stages

    @property
    def crystals(self) -> list[ase.Atoms]:
        """
        List of all atoms of all stages, ordered with the stage ids.
        """
        crystals = []
        for stage_results in self.stages:
            crystals.extend(stage_results.crystals)
        return crystals

    @property
    def key_value_pairs(self) -> list[dict[str, Any]]:
        """
        List of all key value pairs of all stages, ordered the same way as 
        :attr:`RunResults.crystals`.
        """
        key_value_pairs = []
        for stage_results in self.stages:
            key_value_pairs.extend(stage_results.key_value_pairs)
        return key_value_pairs

    def __load_from_files(
        self, run_dir: str
    ) -> tuple[list[StageResults], dict | None]:
        run_info_path = os.path.join(run_dir, "run_info.json")
        if not os.path.exists(run_info_path):
            warnings.warn(f"Could not find run_info.json in {run_dir}.")
            run_info = None
        else:
            with open(os.path.join(run_dir, "run_info.json"), 'r') as f:
                run_info = json.load(f)

        stages = []
        n_stages=0
        stage_id=1
        while True:
            stage_file_path = os.path.join(run_dir, f"stage_{stage_id}.json")
            if os.path.exists(stage_file_path):
                stage = StageResults(run_dir, stage_id)
                stages.append(stage)
                n_stages+=1
                stage_id+=1
            else:
                print(f"Found {n_stages} stages at {run_dir}.")
                break

        if len(stages) == 0:
            warnings.warn(
                f"Could not find any stages in {run_dir}."
            )
        sorted_stages = sorted(stages, key=lambda x: x.id)

        return sorted_stages, run_info


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

    run = StageResults(run_dir, 1)

    print(run.mutations.loc[0, "results"])
