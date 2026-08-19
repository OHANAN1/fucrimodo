import re
from collections.abc import Sequence
from io import StringIO

import ase
import numpy as np

from ..core import Individual
from ..core.abstracts import FitnessFunction
from . import fitness_functions as ff
from .fitness_functions import FitnessFunction
from .global_soap_target import GlobalSOAP


def get_soap_similarity_fitness_list(
    target_soap_features: np.ndarray,
    soap_object: GlobalSOAP,
    rbf_gammas: Sequence[float | int] = (1.0, 0.1, 0.01),
    function_titles: Sequence[str] = (
        "soap_similarity_strong",
        "soap_similarity_mid",
        "soap_similarity_weak",
    ),
    round_result: None | int = None,
    n_jobs: int = 1,
) -> list[FitnessFunction]:
    """Create a list of SOAP RBF similarity fitness functions.

    Build one :class:`SoapRbfSimilarityFitness` instance for each value in
    ``rbf_gammas``, pairing it with the corresponding title from
    ``function_titles``.

    :param target_soap_features: Target SOAP feature vector to compare against.
    :param soap_object: SOAP descriptor object used to compute candidate features.
    :param rbf_gammas: Sequence of RBF gamma values, one per fitness function.
    :param function_titles: Titles for the generated fitness functions.
    :param round_result: Number of decimal places to round results to,
        or ``None`` for no rounding.
    :param n_jobs: Number of parallel jobs passed to each fitness function.
    :returns: List of configured SOAP RBF similarity fitness functions.
    :raises AssertionError: If ``len(function_titles)`` does not equal
        ``len(rbf_gammas)``.
    """

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
    """Create species-pair-specific SOAP RBF similarity fitness functions.

    Build a :class:`SpeciesSpecificSoapRbfSimFitness` instance for every
    unique pair of species (including self-pairs), using the same RBF gamma
    for all functions.
    Unique database titles will be automatically generated based on the
    species pair.

    :param target_soap_features: Target SOAP feature vector to compare against.
    :param species: List of species identifiers, given as atomic numbers
        (``int``) or chemical symbols (``str``).
    :param soap_object: SOAP descriptor object used to compute candidate features.
    :param rbf_gamma: RBF gamma value shared by all generated fitness functions.
    :param function_title: Base title used to build the database title for
        each species pair.
    :param round_result: Number of decimal places to round results to,
        or ``None`` for no rounding.
    :param n_jobs: Number of parallel jobs passed to each fitness function.

    :returns: List of species-pair SOAP RBF similarity fitness functions.

        >('>
                           >('>
    """
    from ase.data import chemical_symbols

    # Convert the soap species to chemical symbols
    species_str = []
    for s in species:
        if type(s) == int:
            s = chemical_symbols[s]
        species_str.append(s)

    # Set up the fitness functions:
    species_specific_fitnesses = []
    for i in range(len(species)):
        for j in range(i, len(species)):
            soap_fit_spec = ff.SpeciesSpecificSoapRbfSimFitness(
                target_soap_features=target_soap_features,
                soap_object=soap_object,
                species=(species_str[i], species_str[j]),
                db_title=f"{function_title}_{species_str[i]}_{species_str[j]}",
                rbf_gamma=rbf_gamma,
                round_result=round_result,
                n_jobs=n_jobs,
            )
            species_specific_fitnesses.append(soap_fit_spec)

    return species_specific_fitnesses


class LegacyRNGAdapter:
    """Wrap a :class:`numpy.random.Generator` to emulate ``np.random.RandomState``.

    This adapter forwards the most common ``RandomState`` methods
    (:meth:`random`, :meth:`choice`, :meth:`normal`, :meth:`randint`) to the
    underlying generator and delegates any other attribute lookups to it as
    well.

    .. note::

        This is a best-effort adapter. Some ``RandomState`` methods (e.g.
        ``rand``, ``randn``, ``seed``, ``random_sample``) are not available on
        a :class:`numpy.random.Generator` and will raise
        :exc:`AttributeError` if called.

    :param rng: The NumPy random generator to wrap.
    :type rng: :class:`numpy.random.Generator`
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

    target_individual = Individual.from_ase(target_structure)

    return target_individual


def get_n_atoms_from_additional_notes(
    additional_notes: str,
) -> int:
    """Load the target structure from the additional notes of the input file.

    It is assumed that the number of atoms is stored as 'Number of atoms: ...'
    in the additional notes string.
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
