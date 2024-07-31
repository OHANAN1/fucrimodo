from abc import ABC, abstractmethod

class BreakCondition(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def check(self, population: list, generation_index: int) -> bool:
        pass

    @abstractmethod
    def __repr__(self):
        class_name = self.__class__.__name__
        variables = vars(self)
        variables_str = ', '.join(
            f'{key}={value}' for key, value in variables.items())
        return f'{class_name}({variables_str})'
