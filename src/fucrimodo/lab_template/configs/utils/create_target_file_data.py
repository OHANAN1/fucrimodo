import os

import ase
import click
from ase.io import read

from fucrimodo.core import Individual
from fucrimodo.customs.global_soap_target import GlobalSOAP
from fucrimodo.utils.target_file_parser import save_to_target_file


def main(atoms_path: str, save_path: str, verbose: bool) -> None:
    if os.path.isfile(save_path):
        raise click.ClickException("FileExistsError: The file already exists.")

    atoms = read(atoms_path)
    assert isinstance(atoms, ase.Atoms)
    ind = Individual.from_ase(atoms)

    # Create the SOAP descriptor object
    soap_kwargs = {
        "r_cut": 15.0,
        "n_max": 8,
        "l_max": 8,
        "sigma": 0.5,
        "species": ind.get_chemical_symbols(),
        "periodic": True,
        "average": "inner",
    }
    descriptor_name = "GlobalSOAP"
    soap = GlobalSOAP(**soap_kwargs)

    # Calculate the feature vector
    target_features = soap.create(ind)

    # Add additional notes about the atoms object
    notes = "Information:\n"
    notes += f"Number of atoms: {len(atoms)}\n"
    notes += f"Chemical formula: {atoms.get_chemical_formula()}\n"
    notes += f"Cell volume: {atoms.get_volume()}\n"
    notes += f"PBC: {atoms.get_pbc()}\n"

    save_to_target_file(
        target_features,
        descriptor_name=descriptor_name,
        descriptor_parameters=soap_kwargs,
        additional_notes=notes,
        save_path=save_path,
    )
