from abc import ABC, abstractmethod
from .population import Population
from ase.db.core import Database
from deap import tools
import numpy as np
import ase
from typing import Callable

class Stage(ABC):
    def __init__(self, id: int):
        self._id = id

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
