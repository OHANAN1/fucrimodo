import ase
from fucrimodo.core.utils.custom_soap import CustomSOAP

def main(atoms: ase.Atoms) -> tuple[str, list[float], dict, str, str]:
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
    descriptor_name = "CustomSOAP"
    soap = CustomSOAP(**kwargs)

    # Calculate the feature vector
    target_features = soap.create(atoms)

    # Create the target file name
    target_file_name = f"{atoms.get_chemical_formula()}_target_file.json"

    # Add additional notes about the atoms object
    notes = "Information:\n"
    notes += f"Number of atoms: {len(atoms)}\n"
    notes += f"Chemical formula: {atoms.get_chemical_formula()}\n"
    notes += f"Cell volume: {atoms.get_volume()}\n"
    notes += f"PBC: {atoms.get_pbc()}\n"

    return descriptor_name, target_features.tolist(), kwargs, notes, target_file_name
