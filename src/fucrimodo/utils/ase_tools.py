import os
import warnings
from typing import Any

import ase
from ase import db
from ase.db.core import Database


def connect_to_existing_database(database_path: str) -> Database:
    """
    Connects to the database and returns the database object.

    If the database does not exist an error is raised,
    to prevent the creation of a new database.
    """
    if not os.path.exists(database_path):
        raise FileNotFoundError("Database not found at {}".format(database_path))
    else:
        database = db.connect(database_path)

    return database


def get_all_structures_from_database(database: Database) -> list[ase.Atoms]:
    """
    Returns a list of all structures in the database.
    """
    structures = []
    for row in database.select():
        structure = row.toatoms()
        structures.append(structure)

    return structures


def get_structures_and_key_value_pairs_from_database(
    database: Database,
) -> tuple[list[ase.Atoms], list[dict]]:
    """
    Returns a list of structures with the key-value pair from the database.
    Return:
    :structures: list[ase.Atoms]
    :key_value_pairs_list: list[dict]
    """
    structures = []
    key_value_pairs_list = []
    for row in database.select():
        structure = row.toatoms()
        structures.append(structure)

        key_value_pairs_list.append(row.key_value_pairs)

    return structures, key_value_pairs_list


def get_unique_keys_of_db(database: Database) -> list[str]:
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
            warnings.warn(f"Could not find key {key} in row {row}.")

    for row in crystals_db.select(filter=filter_stage):
        crystals.append(row.toatoms())
        key_value_pairs.append(row.key_value_pairs)

    if len(crystals) == 0:
        raise ValueError(f"Could not find any crystals for {key}.")

    if len(key_value_pairs) == 0:
        raise ValueError(f"Could not find any key value pairs for {key}.")

    return crystals, key_value_pairs
