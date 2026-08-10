from fucrimodo.customs import population_generators as pop_gen
from fucrimodo.core import Individual
from io import StringIO
import re
import ase
import ase
from .global_soap import GlobalSOAP


def create_target_file_data(
    atoms: ase.Atoms, add_to_notes: None | str = None
) -> tuple[str, list[float], dict, str, str]:
    # Create the SOAP descriptor object
    kwargs = {
        "r_cut": 15.0,
        "n_max": 8,
        "l_max": 8,
        "sigma": 0.5,
        "species": atoms.get_chemical_symbols(),
        "periodic": True,
        "average": "inner",
    }
    descriptor_name = "GlobalSOAP"
    soap = GlobalSOAP(**kwargs)

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

    target_individual = pop_gen.convert_ase_atoms_to_individual(target_structure)

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
