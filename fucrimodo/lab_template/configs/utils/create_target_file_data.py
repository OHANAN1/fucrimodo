import ase
from fucrimodo.core.utils.custom_soap import CustomSOAP

def main(atoms: ase.Atoms) -> tuple[list[float], dict, str, str]:
    target_features = []
    kwargs = {}
    notes = ""
    target_file_name = ""

    # Create the SOAP descriptor object
    kwargs = {
        "r_cut": 15.0,
        "n_max": 8,
        "l_max": 8,
        "sigma": 0.5,
        "species": atoms.get_chemical_symbols(),
        "periodic": True,
        "average": "inner"
    }
    soap = CustomSOAP(**kwargs)

    # Calculate the feature vector
    target_features = soap.create(atoms)

    # Create the target file name
    target_file_name = f"{atoms.get_chemical_formula()}_target_file.json"

    # Add additional notes about the atoms object
    notes = f"Number of atoms: {len(atoms)}"
    notes += f"\nChemical formula: {atoms.get_chemical_formula()}"
    notes += f"\nCell volume: {atoms.get_volume()}"
    notes += f"\nPBC: {atoms.get_pbc()}"
    notes += f"\nPositions: {atoms.get_positions()}"
    notes += f"\nCell: {atoms.get_cell()}"
    notes += f"\nAtomic numbers: {atoms.get_atomic_numbers()}"

    return target_features.tolist(), kwargs, notes, target_file_name
