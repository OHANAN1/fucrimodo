import numpy as np
from ase.db.core import Database

from typing import Any, Optional
from dscribe.descriptors import SOAP
import os
from ase import db
import ase
from ase.data import atomic_numbers
from numpy.typing import NDArray
from typing import Sequence
from sklearn.metrics.pairwise import cosine_similarity
import warnings 


def connect_to_existing_database(database_path: str) -> Database:
    """
    Connects to the database and returns the database object.

    If the database does not exist an error is raised,
    to prevent the creation of a new database.
    """
    if not os.path.exists(database_path):
        raise FileNotFoundError(
            "Database not found at {}".format(database_path))
    else:
        database = db.connect(database_path)

    return database


def get_unique_atom_types_of_db(
    database: Database,
    verbose: int = 1
) -> list[str]:
    if verbose > 0:
        print()
        print("Getting unique atom types of db...")

    all_atom_types = np.array([])

    for row in database.select():
        atom_types = row.symbols
        all_atom_types = np.append(all_atom_types, atom_types)

    unique_atom_types = np.unique(all_atom_types)
    if verbose > 0:
        print("Unique atom types of db: {}".format(unique_atom_types))

    return unique_atom_types.tolist()


def get_unique_atomic_numbers_of_db(
    database: Database,
    verbose: int = 1
) -> list[int]:
    if verbose > 0:
        print()
        print("Getting unique atomic numbers of db...")

    unique_atom_types = get_unique_atom_types_of_db(database, verbose=verbose)

    unique_atomic_numbers = [
        atomic_numbers[element] for element in unique_atom_types
    ]

    if verbose > 0:
        print("Unique atomic numbers of db: {}".format(unique_atomic_numbers))

    return unique_atomic_numbers


def create_soap_obj_from_database(
    database_path: str,
    r_cut: Optional[float] = None,
    n_max: Optional[float] = None,
    l_max: Optional[float] = None,
    sigma: float = 1,
    rbf: str = "gto",
    weighting: Optional[dict] = None,
    average: str = "off",
    compression: dict = {"mode": "off", "species_weighting": None},
    periodic: bool = False,
    sparse: bool = False,
    dtype: str = "float64",
    verbose: int = 1
) -> SOAP:
    """
    Creates a SOAP object based on the unique atom types of the database.
    All other parameters can be set manually.
    The default values are the same as in the DScribe documentation.

    Note: Set Periodic to True if the database contains periodic crystals!
    """
    print("Creating SOAP object with specified parameters...")
    database = connect_to_existing_database(database_path)

    species = get_unique_atom_types_of_db(database, verbose=verbose)
    print("Unique atom types of db: {}".format(species))

    return SOAP(
        species=species,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        sigma=sigma,
        rbf=rbf,
        weighting=weighting,
        average=average,
        compression=compression,
        periodic=periodic,
        sparse=sparse,
        dtype=dtype
    )


def get_all_crystals_from_database(database: Database) -> list[ase.Atoms]:
    """
    Returns a list of all crystals in the database.
    """
    crystals = []
    for row in database.select():
        crystal = row.toatoms()
        crystals.append(crystal)

    return crystals


def get_soap_features_from_db(
    database: Database,
    soap: SOAP,
) -> Sequence[NDArray[np.float64]]:
    """
    Returns a list of all SOAP features of the database.
    Manly used for type hinting.
    (Soap features are than only NDArrays)
    """
    crystal_list = get_all_crystals_from_database(database)
    return soap.create(crystal_list)  # type: ignore


def get_crystals_and_key_value_pairs_from_database(
    database: Database,
) -> tuple[list[ase.Atoms], list[dict]]:
    """
    Returns a list of crystals with the key-value pair from the database.
    Return:
    :crystals: list[ase.Atoms]
    :key_value_pairs_list: list[dict]
    """
    crystals = []
    key_value_pairs_list = []
    for row in database.select():
        crystal = row.toatoms()
        crystals.append(crystal)

        key_value_pairs_list.append(row.key_value_pairs)

    return crystals, key_value_pairs_list


def get_data_of_crystal_at_id(
    database: Database,
    crystal_id: int,
    soap_obj: SOAP | None = None
) -> tuple[ase.Atoms, dict, NDArray[np.float64] | None]:
    """
    Returns the crystal, key_value_pairs of the crystal with the specified id.
    """
    row = list(database.select(selection=crystal_id))[0]
    crystal = row.toatoms()
    key_value_pairs = row.key_value_pairs

    if soap_obj is not None:
        soap_features = soap_obj.create([crystal])
        return crystal, key_value_pairs, soap_features  # type: ignore

    else:
        return crystal, key_value_pairs, None


def exclude_structuraly_similar_crystals(
    crystals: list[ase.Atoms],
    key_value_pairs_list: list[dict],
    compared_crystal: ase.Atoms,
) -> tuple[list[ase.Atoms], list[dict]]:
    filtered_crystals = []
    filtered_key_value_pairs_list = []
    for i, crystal in enumerate(crystals):
        are_equal = True
        if len(crystal) == len(compared_crystal):
            are_equal = False
            if (crystal.positions != compared_crystal.positions).any():
                are_equal = False

            atomic_numbers = crystal.get_atomic_numbers()
            compared_atomic_numbers = compared_crystal.get_atomic_numbers()
            if (atomic_numbers != compared_atomic_numbers).any():
                are_equal = False

        else:
            are_equal = False

        if not are_equal:
            filtered_crystals.append(crystal)
            filtered_key_value_pairs_list.append(key_value_pairs_list[i])

    return filtered_crystals, filtered_key_value_pairs_list


def remove_same_key_value_pairs(
    key_value_pairs_list: list[dict],
    crystals: list[ase.Atoms],
    key_value_pairs_to_check: dict
) -> tuple[list[ase.Atoms], list[dict]]:
    """
    Removes crystals with the same key value pairs as the specified
    key value pairs.
    """
    filtered_crystals = []
    filtered_key_value_pairs_list = []
    for i, key_value_pairs in enumerate(key_value_pairs_list):
        is_equal = True
        for key in key_value_pairs_to_check.keys():
            if key_value_pairs[key] != key_value_pairs_to_check[key]:
                is_equal = False
                break

        if not is_equal:
            filtered_crystals.append(crystals[i])
            filtered_key_value_pairs_list.append(key_value_pairs_list[i])

    return filtered_crystals, filtered_key_value_pairs_list


def remove_similar_crystals(
    crystals: list[ase.Atoms],
    key_value_pairs_list: list[dict],
    soap: SOAP,
    similarity_threshold: float = 0.99
) -> tuple[list[ase.Atoms], list[dict]]:
    """
    Removes similar crystals from the list of crystals.
    """
    if len(crystals) == 0:
        raise ValueError("No crystals left to compare!")
    elif len(crystals) == 1:
        soap_features = [soap.create(crystals)]
    else:
        soap_features = soap.create(crystals)

    similarity_matrix = cosine_similarity(soap_features, soap_features)
    similarities = [similarity[0] for similarity in similarity_matrix]

    filtered_crystals = []
    filtered_key_value_pairs_list = []
    for i, similarity in enumerate(similarities):
        if similarity < similarity_threshold:
            filtered_crystals.append(crystals[i])
            filtered_key_value_pairs_list.append(key_value_pairs_list[i])

    return filtered_crystals, filtered_key_value_pairs_list


def get_filtered_data_from_db(
    database: Database,
    exclude_crystal_with_id: int,
    keys_to_check: list[str] | None = None,
    remove_crystals_similar_to_excluded: bool = True,
    similarity_threshold: float = 0.99,
    soap: SOAP | None = None
) -> tuple[list[ase.Atoms], list]:
    """
    Returns the data of the database,
    excluding the crystal with the specified id.

    The data is a tuple of the form:
    raw_crystals, raw_energies

    The raw crystals/energies explicitly exclude the crystal/energy
    specified in the class init.
    the target crystal is set by the target_crystal_id from the class.

    The resulting crystals are also compared to ensure that the
    excluded crystal is not included in the list of crystals.
    """
    print()
    print("Getting filtered data from database...")
    excluded_crystal, excluded_key_val_pairs = get_data_of_crystal_at_id(
        database, exclude_crystal_with_id)

    def filter_specified_crystal(row):
        return row.get("id") != exclude_crystal_with_id

    crystals = []
    key_value_pairs_list = []
    for row in database.select(filter=filter_specified_crystal):
        crystals.append(row.toatoms())
        key_value_pairs_list.append(row.key_value_pairs)

    filtered_data = exclude_structuraly_similar_crystals(
        crystals, key_value_pairs_list, excluded_crystal
    )
    filtered_crystals, filtered_key_val_pairs_list = filtered_data

    if keys_to_check is not None:
        for key in keys_to_check:
            filtered_data = remove_same_key_value_pairs(
                filtered_key_val_pairs_list,
                filtered_crystals,
                {key: excluded_key_val_pairs[key]}
            )
            filtered_crystals, filtered_key_val_pairs_list = filtered_data

    if remove_crystals_similar_to_excluded:
        if soap is None:
            raise ValueError(
                "Soap object must be specified to remove similar crystals!")
        filtered_data = remove_similar_crystals(
            filtered_crystals, filtered_key_val_pairs_list,
            soap, similarity_threshold
        )
        filtered_crystals, filtered_key_val_pairs_list = filtered_data

    print("Filtered data from database.")
    print("Number of crystals before: {}".format(len(crystals)))
    print("Number of crystals now: {}".format(len(filtered_crystals)))

    return filtered_crystals, filtered_key_val_pairs_list


def get_unique_keys_of_db(
    database: Database
) -> list[str]:
    keys = set()
    for row in database.select():
        key_value_pairs = row.key_value_pairs
        for key in key_value_pairs:
            keys.add(key)
    return list(keys)


def get_data_with_specific_key_value_from_db(
    crystals_db: Database, key: str, value: Any
) -> tuple[list[ase.Atoms], list[dict[str, Any]]]:
    """
    Returns the crystals and key value pairs from the database that have the
    specified key and value.
    """
    crystals = []
    key_value_pairs = []

    def filter_stage(row):
        if hasattr(row, "key_value_pairs"):
            if "stage_id" in row.key_value_pairs.keys():
                return row.key_value_pairs[key] == value
            else:
                return False
        elif hasattr(row, "is_target"):
            return False

        else:
            warnings.warn(
                f"Could not find key {key} in row {row}."
            )

    for row in crystals_db.select(filter=filter_stage):
        crystals.append(row.toatoms())
        key_value_pairs.append(row.key_value_pairs)

    if len(crystals) == 0:
        raise ValueError(
            f"Could not find any crystals for {key}."
        )

    if len(key_value_pairs) == 0:
        raise ValueError(
            f"Could not find any key value pairs for {key}."
        )

    return crystals, key_value_pairs
