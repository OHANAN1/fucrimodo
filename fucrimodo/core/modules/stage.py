from abc import ABC, abstractmethod
from .population import Population
from ase.db.core import Database
from deap import tools

class Stage(ABC):
    """Abstract base class for stages in the optimization algorithm.

    Stages are used to define the different steps in the optimization
    algorithm. Each stage can perform a defined optimization algorithm, for
    example a genetic algorithm or a swarm search algorithm.
    Stages should not be run directly, but should be used with the 
    multi-stage optimization algorithm.

    :param name: User friendly name of the stage, used for analysis.
    :param description: Optional description of the stage, used for analysis.
    """
    def __init__(self, name: str, description: str) -> None:
        self._name = name
        self._description = description

    @abstractmethod
    def run(
        self, 
        population: Population,
        global_log: tools.Logbook,
        global_stats: tools.MultiStatistics | None,
    ) -> Population:
        pass

    @abstractmethod
    def save_results(self, save_dir: str, crystals_db: Database):
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
        """Return the type of the stage.

        This method is used to determine the type of the stage in the analysis
        scripts.
        """
        return self.__class__.__name__

    @property
    @abstractmethod
    def info_dict(self) -> dict:
        """Contains all information about the stage that is necessary to
        recreate the stage (manually or automatically).

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
        """Optional description of the stage, used for analysis."""
        return self._description
