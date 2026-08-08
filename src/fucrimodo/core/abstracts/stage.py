from abc import ABC, abstractmethod
from datetime import datetime
import os

from ..individual import Individual
from ..population import Population
from ase.db.core import Database
from deap import tools
from typing import Callable
import logging


class Stage(ABC):
    """Abstract base class for stages in the optimization algorithm.

    Stages are the building blocks of the multi-stage optimization algorithm.
    Each stage can perform a defined optimization algorithm, for example a
    genetic algorithm or a swarm search. Stages should not be run directly, but
    should be executed by the class :class:`MultiStageSearch`.

    :param name: User friendly name of the stage, used for analysis.
    :param description: Optional description of the stage, used for analysis.
    """

    def __init__(self, name: str, description: str) -> None:
        self._name = name
        self._description = description

    @property
    def start_time(self) -> datetime:
        """Property to note when stage was started.

        The :class:`MultiStageSearch` class automatically sets the time.
        To set manually, please use the method :meth:`set_start_time` to set
        time to current time.
        """
        if not hasattr(self, "_start_time"):
            raise AttributeError(
                f"{self.__class__.__name__}: No start time set. "
                "Please set a start time with the set_start_time method."
            )
        return self._start_time

    def set_start_time(self):
        """Set the start time of the stage to the current time.

        Will be set automatically before the run method by the
        :class:`MultiStageSearch`.
        """
        self._start_time = datetime.now()

    @property
    def end_time(self) -> datetime:
        """Return the end time of the stage.

        If the end time is not set, the start time is returned.
        Will be set automatically after the run by the
        :class:`MultiStageSearch`.
        """
        if not hasattr(self, "_end_time"):
            return self.start_time
        return self._end_time

    def set_end_time(self):
        self._end_time = datetime.now()

    @property
    def logger(self) -> logging.Logger:
        """Logger of the stage.

        The :class:`MultiStageSearch` class automatically sets the appropriate
        logger.
        """
        if not hasattr(self, "_logger"):
            raise AttributeError(
                f"{self.__class__.__name__}: No logger set. Please set a logger."
            )
        return self._logger

    @logger.setter
    def logger(self, value):
        self._logger = value

    @property
    def stage_dir(self) -> str:
        """Directory to store information about the run.

        The :class:`MultiStageSearch` class automatically assigns the
        appropriate directory.
        """
        if not hasattr(self, "_stage_dir"):
            raise AttributeError(
                f"{self.__class__.__name__}: No stage directory set. Please set a stage directory."
            )
        return self._stage_dir

    @stage_dir.setter
    def stage_dir(self, value: str):
        # Create dir if it not already exists
        if not os.path.isdir(value):
            os.mkdir(value)
        self._stage_dir = value

    @abstractmethod
    def run(
        self,
        population: Population,
        global_log: tools.Logbook,
        global_stats: tools.MultiStatistics | None,
    ) -> Population:
        pass

    @abstractmethod
    def save_results(
        self,
        save_dir: str,
        structures_db: Database,
        global_statistics_dict: dict[str, Callable[[Individual], float]] | None = None,
    ) -> None:
        """Method to save the results of the stage to a given directory and
        ASE database.

        This method should save the results of the optimization algorithm so it
        can be analyzed later. How the results are saved is up to the
        implementation of the stage.
        If necessary the analysis scripts must be adjusted to read the saved
        results.
        """
        pass

    def type(self) -> str:
        """Return the type of the stage, i.e. class name.

        This method is used to determine the type of the stage in the analysis
        scripts.
        """
        return self.__class__.__name__

    @property
    @abstractmethod
    def info_dict(self) -> dict:
        """Contains all information about the stage that is necessary to
        recreate the stage.

        Information must be savable to a JSON file.
        What information is saved is up to the implementation of the stage.
        If necessary the analysis scripts must be adjusted to read the saved
        information.
        The :class:`MultiStageSearch` algorithm extends this dictionary with
        the :attr:`id`, :attr:`type`, :attr:`name` and :attr:`description`.
        """
        pass

    @property
    def id(self) -> int:
        """Unique identifier for the stage.

        Is set automatically by the MultiStageSearch algorithm.
        """
        if not hasattr(self, "_id"):
            raise AttributeError(
                f"{self.__class__.__name__}: Please manually set ID before accessing it."
            )
        return self._id

    @id.setter
    def id(self, value):
        self._id = value

    @property
    def name(self) -> str:
        """User friendly name of the stage, used for analysis."""
        return self._name

    @property
    def description(self) -> str:
        """Optional description of the stage, used to identify it."""
        return self._description
