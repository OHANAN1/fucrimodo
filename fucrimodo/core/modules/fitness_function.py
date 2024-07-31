import ase
from abc import ABC, abstractmethod

# ╔══════════════════════════════════════════════════════════╗
# ║        Abstract Base Class for Fitness Functions         ║
# ╚══════════════════════════════════════════════════════════╝


class FitnessFunction(ABC):
    """
    Abstract class for fitness functions.
    The fitness function should take a array of crystal structures
    as ases.Atoms objects and return a list of floats.
    The return is a list of fitness values for each crystal structure.

    The class is callable and can be used like this:

    fitness_function = FitnessFunction(...)
    fitness = fitness_function(list[Individual])
    """

    def __init__(self):
        pass

    @abstractmethod
    def evaluate_individual(self, individual: ase.Atoms) -> float:
        """
        This function takes a single individual and returns a fitness value.
        """
        pass

    @abstractmethod
    def adjust_to_population(
        self,
        population: list[ase.Atoms]
    ) -> None:
        """
        Optional function that can be used to adjust for example the
        rbf gamma value to the given population.
        If not needed it can be left empty.
        """
        pass

    @abstractmethod
    def set_db_title(self, title: str) -> None:
        """
        Optional function that can be used to set the title of the
        ase database. Therefore no space, no numbers and no special
        characters are allowed. '_' is allowed.
        """
        forbidden_chars = [
            " ", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
        ]
        if any([char in title for char in forbidden_chars]):
            raise ValueError(
                "The title of the database must not contain any spaces, "
                "numbers or special characters."
                f"Given title: {title}"
            )

        self.db_title = title

    @abstractmethod
    def get_db_title(self) -> str:
        """
        Optional function that can be used to get the title of the
        ase database.
        """
        pass

    @abstractmethod
    def __repr__(self):
        class_name = self.__class__.__name__
        variables = vars(self)
        variables_str = ""
        for key, value in variables.items():
            if key == "target_soap_features":
                value = "target_soap_features"
            variables_str += f"{key}={value}, "

        return f'{class_name}({variables_str})'
