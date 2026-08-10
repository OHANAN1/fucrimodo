from collections.abc import Sequence

import ase
import numpy as np
import re
from io import StringIO

from ..core.abstracts import FitnessFunction
from ..core import Individual
from . import fitness_functions as ff
from .fitness_functions import FitnessFunction
from .global_soap_target import GlobalSOAP


def get_soap_similarity_fitness_list(
    target_soap_features: np.ndarray,
    soap_object: GlobalSOAP,
    rbf_gammas: Sequence[float | int] = [1.0, 0.1, 0.01],
    function_titles: list[str] = [
        "soap_similarity_strong",
        "soap_similarity_mid",
        "soap_similarity_weak",
    ],
    round_result: None | int = None,
    n_jobs: int = 1,
) -> list[FitnessFunction]:

    assert len(function_titles) == len(
        rbf_gammas
    ), "Define same number of titles as rbf gammas."

    soap_fitnesses = []
    for gamma, title in zip(rbf_gammas, function_titles):
        soap_fitnesses.append(
            ff.SoapRbfSimilarityFitness(
                target_soap_features=target_soap_features,
                soap_object=soap_object,
                rbf_gamma=gamma,
                db_title=title,
                round_result=round_result,
                n_jobs=n_jobs,
            )
        )

    return soap_fitnesses


def get_species_specific_soap_sim_fitness_list(
    target_soap_features,
    species: list[str | int],
    soap_object: GlobalSOAP,
    rbf_gamma: int | float = 0.1,
    function_title: str = "species_specific_fit",
    round_result: None | int = None,
    n_jobs: int = 1,
) -> list[FitnessFunction]:
    from ase.data import chemical_symbols

    # Convert the soap species to chemical symbols
    soap_species_sym = []
    for s in species:
        if type(s) == int:
            s = chemical_symbols[s]
        soap_species_sym.append(s)

    # Set up the fitness functions:
    species_specific_fitnesses = []
    for i in range(len(species)):
        for j in range(i, len(species)):
            soap_fit_spec = ff.SpeciesSpecificSoapRbfSimFitness(
                target_soap_features=target_soap_features,
                soap_object=soap_object,
                species=(soap_species_sym[i], soap_species_sym[j]),
                db_title="{}_{}_{}".format(function_title, species[i], species[j]),
                rbf_gamma=rbf_gamma,
                round_result=round_result,
                n_jobs=n_jobs,
            )
            species_specific_fitnesses.append(soap_fit_spec)

    return species_specific_fitnesses


def convert_ase_atoms_to_individual(atoms: ase.Atoms) -> Individual:
    return Individual(
        positions=atoms.get_positions(),
        cell=atoms.get_cell(),
        pbc=atoms.pbc,
        symbols=atoms.get_chemical_symbols(),
    )


class LegacyRNGAdapter:
    """Wrap a numpy.random.Generator so it looks like np.random.RandomState.

    This can be used whenever the legacy np.random api needs to be used.
    """

    def __init__(self, rng: np.random.Generator):
        self._rng = rng

    def random(self, *args, **kwargs):
        return self._rng.random(*args, **kwargs)

    def choice(self, *args, **kwargs):
        return self._rng.choice(*args, **kwargs)

    def normal(self, loc=0.0, scale=1.0, size=None):
        return self._rng.normal(loc=loc, scale=scale, size=size)

    def randint(self, low, high=None, size=None, dtype=int):
        if high is None:
            return self._rng.integers(low, size=size, dtype=dtype)
        return self._rng.integers(low, high, size=size, dtype=dtype)

    def __getattr__(self, name):
        # Forward anything else to the Generator
        return getattr(self._rng, name)


def get_target_individual_from_additional_notes(
    additional_notes: str,
) -> Individual:
    """Load the target structure from the additional notes of the input file.

    It is assumed that the target structure is stored in the additional notes
    as a CIF string. The CIF string is extracted from the additional notes
    using a regex pattern. The CIF string is then loaded into an ASE Atoms
    object.
    """
    regex = r"CIF:(.*)"
    match = re.search(regex, additional_notes)
    if match:
        # If there is a match, return the volume
        cif_string = str(match.group(1))
    else:
        raise ValueError(
            f"Could not find the pattern in the additional notes"
            f" with the regex pattern {regex}. \n"
            f"Additional notes: {additional_notes}"
        )

    cif_string = cif_string.replace("NEWLINE", "\n")
    cif_string = cif_string.replace("QUOTATION_MARK", '"')
    with StringIO(cif_string) as f:
        from ase.io import read

        target_structure = read(f, format="cif")

    assert (
        type(target_structure) is ase.Atoms
    ), "Please verify that CIF-string really is ase.Atoms object!"

    target_individual = convert_ase_atoms_to_individual(target_structure)

    return target_individual


def get_n_atoms_from_additional_notes(
    additional_notes: str,
) -> int:
    """Load the target structure from the additional notes of the input file.

    It is assumed that the target structure is stored in the additional notes
    as a CIF string. The CIF string is extracted from the additional notes
    using a regex pattern. The CIF string is then loaded into an ASE Atoms
    object.
    """
    regex = r"Number of atoms:(.*)"
    match = re.search(regex, additional_notes)
    if match:
        # If there is a match, return the volume
        n_atoms = int(match.group(1))
    else:
        raise ValueError(
            f"Could not find the pattern in the additional notes"
            f" with the regex pattern {regex}. \n"
            f"Additional notes: {additional_notes}"
        )

    return n_atoms
