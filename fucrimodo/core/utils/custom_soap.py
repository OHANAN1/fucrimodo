from dscribe.descriptors import SOAP
import numpy as np
from numpy.typing import NDArray
import ase
import warnings

class CustomSOAP():
    def __init__(
        self,
        r_cut: float,
        n_max: int,
        l_max: int,
        species: list[str] | list[int],
        average: str = 'inner',
        sigma: float = 1.0,
        periodic: bool = True,
    ) -> None:

        self.r_cut = r_cut
        self.n_max = n_max
        self.l_max = l_max
        self.sigma = sigma
        self.periodic = periodic
        self.average = average

        # Only temporary solution to make species always a list of strings
        # I need to figure out how to handle this better
        if all(isinstance(s, int) for s in species):
            from ase.data import chemical_symbols
            str_species = [chemical_symbols[s] for s in species]
            self._species = str_species
        elif all(isinstance(s, str) for s in species):
            str_species = []
            for s in species:
                if not isinstance(s, str):
                    raise ValueError('Species must be a list of strings or integers')
                str_species.append(s)
            self._species = str_species
        else:
            raise ValueError('Species must be a list of strings or integers')

        self._dscribe_soap = SOAP( 
            r_cut = r_cut,
            n_max = n_max,
            l_max = l_max,
            species = species,
            sigma = sigma,
            periodic = periodic,
            average = average,
            sparse = False
        )

    @property
    def species(self) -> list[str]:
        """List of chemical symbols used to calculate the SOAP descriptor."""
        return self._species

    def get_init_params(self) -> dict:
        """
        Returns a dictionary with parameters that where used to set up the 
        class.

        Example:

        soap_params = custom_soap.get_init_params()
        custom_soap_copy = CustomSOAP(**soap_params)
        """
        return {
            "r_cut": self.r_cut,
            "n_max": self.n_max,
            "l_max": self.l_max,
            "species": self.species,
            "sigma": self.sigma,
            "periodic": self.periodic,
            "average": self.average
        }

    def __is_valid(self, crystal: ase.Atoms) -> bool:
        if not isinstance(crystal, ase.Atoms):
            warnings.warn('Input is not an ASE Atoms object')
            return False

        atomic_numbers = crystal.get_atomic_numbers()
        if np.isnan(atomic_numbers).any():
            warnings.warn('Atomic numbers contain NaNs')
            return False

        positions = crystal.get_positions()
        if np.isnan(positions).any():
            warnings.warn('Positions contain NaNs')
            return False

        if len(atomic_numbers) != len(positions):
            warnings.warn(
                'Number of atomic numbers and positions do not match'
            )
            return False

        if len(atomic_numbers) == 0:
            warnings.warn('No atoms in the crystal')
            return False

        if not hasattr(crystal, 'cell'):
            warnings.warn('Cell is not defined')
            return False

        cell = crystal.get_cell()[:] # type: ignore
        if np.nan in cell:
            warnings.warn('Cell contains NaNs')
            return False

        return True

    def get_number_of_features(self) -> int:
        return self._dscribe_soap.get_number_of_features()

    def get_location(self, species: tuple) -> slice:
        return self._dscribe_soap.get_location(species)

    def create(
        self,
        system: list[ase.Atoms] | ase.Atoms,
        n_jobs=1,
        only_physical_cores=False,
        verbose=False
    ) -> NDArray[np.float64]:
        if isinstance(system, ase.Atoms):
            if not self.__is_valid(system):
                raise ValueError('Invalid input. Check warnings for details.')
        else:
            if not all(self.__is_valid(crystal) for crystal in system):
                raise ValueError('Invalid input. Check warnings for details.')

        try:
            return self._dscribe_soap.create(  # type: ignore
                system=system,
                n_jobs=n_jobs,
                only_physical_cores=only_physical_cores,
                verbose=verbose
            )
        except Exception as e:
            raise ValueError(f'Error in creating SOAP: {e}')
