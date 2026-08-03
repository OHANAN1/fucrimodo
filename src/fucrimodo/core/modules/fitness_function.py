from abc import ABC, abstractmethod
from .individual import Individual
from .population import Population


class FitnessFunction(ABC):
    """Evaluate how well individuals fullfill an objective.

    Fitness functions are used for example in the genetic algorithm to
    assign a fitness value to an individual.
    Individuals can then be compared based on their fitness value.

    :param db_title: Descriptive title that can be used to refer to the
        fitness function in an ASE database. See :attr:'db_title' for more
        information.

    :raises ValueError: If the :data:'db_title' contains any spaces, numbers
        or special characters other than '_'.
    """

    def __init__(self, db_title: str | None = None):
        if db_title is not None:
            self.db_title = db_title

    @property
    def db_title(self) -> str:
        """Descriptive title that can be used to refer to the
        fitness function in an ASE database.
        This title must not contain any spaces, numbers or special
        characters. '_' is allowed.

        If the title is not set, the class name of the fitness function
        is used to generate a title.

        :raises ValueError: If set and the title contains any spaces, numbers
            or special characters other than '_'.
        """
        if not hasattr(self, "_db_title"):
            return self.__class__.__name__

        return self._db_title

    @db_title.setter
    def db_title(self, title: str):
        forbidden_chars = [" ", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        if any([char in title for char in forbidden_chars]):
            raise ValueError(
                "The title of the database must not contain any spaces, "
                "numbers or special characters."
                f"Given title: {title}"
            )
        self._db_title = title

    @abstractmethod
    def evaluate_individual(self, individual: Individual) -> float:
        """Method to calculate the fitness value of an individual.

        :param individual: Individual

        :returns: Fitness of individual
        """
        pass

    def evaluate_individuals(self, individuals: list[Individual]) -> list[float]:
        """Method to calculate the fitness value of a list of individuals.

        Sometimes it is more efficient to evaluate multiple individuals at once
        instead of evaluating them one by one. E.g. through parallelization.
        If not implemented by the class that inherits from this class,
        this function will just evaluate each individual one by one.

        :param individuals: List of individuals to evaluate.

        :returns: List of fitness values of each individual.
        """
        fitnesses = []
        for individual in individuals:
            fitness = self.evaluate_individual(individual)
            fitnesses.append(fitness)
        return fitnesses

    def adjust_to_population(self, population: Population) -> None:
        """
        Optional function that can be used to adjust for example the
        rbf gamma value to the given population.
        """
        pass

    def __repr__(self):
        class_name = self.__class__.__name__
        variables = vars(self)
        variables_str_list = []
        for key, value in variables.items():
            if key == "target_soap_features":
                value = "target_soap_features"
            if key.startswith("_"):
                continue
            variables_str_list.append(f"{key}={value}")
        variables_str = ", ".join(variables_str_list)
        return f"{class_name}({variables_str})"
