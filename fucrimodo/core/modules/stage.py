from abc import ABC, abstractmethod
from .population import Population
from ase.db.core import Database
from deap import tools
import numpy as np
import ase
from typing import Callable

class Stage(ABC):
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
    def save_results(self, save_path: str, crystals_db: Database):
        pass

    @property
    def id(self) -> int:
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
