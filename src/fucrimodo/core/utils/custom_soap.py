from dscribe.descriptors import SOAP
import numpy as np
from numpy.typing import NDArray
import ase
import warnings


class CustomSOAP:
    def __init__(
        self,
        r_cut: float,
        n_max: int,
        l_max: int,
        species: list[str] | list[int],
        average: str = "inner",
        sigma: float = 1.0,
        periodic: bool = True,
    ) -> None:
        raise DeprecationWarning(
            "Class moved! Please use: fucrimodo.customs.global_soap_target.GlobalSOAP instead!"
        )
