import random
import ase
from ase.ga.utilities import CellBounds
import numpy as np
from fucrimodo.core.modules import PopulationGenerator
from fucrimodo.core.utils.closest_distances_class import CustomClosestDistances
from fucrimodo.core.utils.cellbounds_custom import CustomCellBounds
from ase.geometry import cell
import warnings
from icecream import ic
from ase.ga.startgenerator import StartGenerator
from ase.ga.utilities import closest_distances_generator
from ase.data import atomic_numbers
from tqdm import tqdm


def crystal_is_valid(
    atoms: ase.Atoms | None,
    cell_bounds: CustomCellBounds,
):
    if atoms is None:
        ic("Atoms object is None")
        return False

    if not isinstance(atoms, ase.Atoms):
        ic("Atoms object is not an ase.Atoms object")
        return False

    if len(atoms) == 0:
        ic("Atoms object has no atoms")
        return False

    if not cell_bounds.is_within_bounds(atoms.cell):
        ic("Cell is not within bounds")
        return False

    if not atoms.pbc.all():
        ic("Atoms object has no periodic boundary conditions")
        return False

    if np.isnan(atoms.get_positions()).any():
        ic("Atoms object has NaN positions")
        return False

    if np.isnan(atoms.get_cell()).any():
        ic("Atoms object has NaN cell")
        return False

    if np.isnan(atoms.get_atomic_numbers()).any():
        ic("Atoms object has NaN atomic numbers")
        return False

    return True


def create_random_atoms_object_2(
    atom_types: list[str],
    cell_bounds: CustomCellBounds,
) -> ase.Atoms | None:

    bounds = cell_bounds.bounds

    cell_pars = []
    for param in ["a", "b", "c"]:
        param_bounds = bounds[param]
        param = np.random.uniform(param_bounds[0], param_bounds[1])
        cell_pars.append(param)

    new_cell = cell.Cell(np.array([[cell_pars[0], 0, 0],
                                   [0, cell_pars[1], 0],
                                   [0, 0, cell_pars[2]]]))
    try:

        atom_typ = [random.choice(atom_types), random.choice(atom_types)]
        scaled_position = [np.random.rand(3), np.random.rand(3)]

        atoms = ase.Atoms(
            atom_typ,
            scaled_positions=scaled_position,
            cell=new_cell,
            pbc=True
        )

        return atoms
    except Exception as e:
        warnings.warn(
            "Error creating random atoms object: {}".format(e), UserWarning
        )
        return None


def create_random_atoms_object(
    atom_types: list[str],
    cell_bounds: CustomCellBounds,
) -> ase.Atoms | None:

    bounds = cell_bounds.bounds

    cell_pars = []
    for param in ["a", "b", "c"]:
        param_bounds = bounds[param]
        param = np.random.uniform(param_bounds[0], param_bounds[1])
        cell_pars.append(param)

    new_cell = cell.Cell(np.array([[cell_pars[0], 0, 0],
                                   [0, cell_pars[1], 0],
                                   [0, 0, cell_pars[2]]]))
    try:

        atom_typ = random.choice(atom_types)
        scaled_position = np.random.rand(3)

        atoms = ase.Atoms(
            atom_typ,
            scaled_positions=[scaled_position],
            cell=new_cell,
            pbc=True
        )

        return atoms
    except Exception as e:
        warnings.warn(
            "Error creating random atoms object: {}".format(e), UserWarning
        )
        return None


def create_one_atomic_crystals(
    atom_types: list[str],
    cell_bounds: CustomCellBounds,
    total_number_of_atoms: int,
):
    print("Creating one atomic crystals...")
    atoms_list = []
    max_steps = 2*total_number_of_atoms
    step = 0
    while len(atoms_list) < total_number_of_atoms:
        print(
            "Adding atoms object {}/{}".format(
                len(atoms_list), total_number_of_atoms), end="\r"
        )
        atoms_object = create_random_atoms_object(
            atom_types, cell_bounds
        )

        if step > max_steps:
            break

        if crystal_is_valid(atoms_object, cell_bounds):
            atoms_list.append(atoms_object)
            step += 1
        else:
            step += 1

    print("Created {} atomic crystals".format(len(atoms_list)))
    print()
    return atoms_list


def create_two_atomic_crystals(
    atom_types: list[str],
    cell_bounds: CustomCellBounds,
    total_number_of_atoms: int,
):
    print("Creating one atomic crystals...")
    atoms_list = []
    max_steps = 2*total_number_of_atoms
    step = 0
    while len(atoms_list) < total_number_of_atoms:
        print(
            "Adding atoms object {}/{}".format(
                len(atoms_list), total_number_of_atoms), end="\r"
        )
        atoms_object = create_random_atoms_object_2(
            atom_types, cell_bounds
        )

        if step > max_steps:
            break

        if crystal_is_valid(atoms_object, cell_bounds):
            atoms_list.append(atoms_object)
            step += 1
        else:
            step += 1

    print("Created {} atomic crystals".format(len(atoms_list)))
    print()
    return atoms_list


if __name__ == "__main__":
    from ase.visualize import view

    atom_types = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"]
    cell_bounds = CustomCellBounds(
        bounds={
            "a": [1, 10],
            "b": [1, 10],
            "c": [1, 10],
        }
    )
    print("Bounds: ", cell_bounds.bounds)
    total_number_of_atoms = 10

    atoms_list = create_one_atomic_crystals(
        atom_types, cell_bounds, total_number_of_atoms
    )

    for atoms in atoms_list:
        print(atoms)
        print(atoms.cell)
        print(atoms.get_positions())
        print(atoms.get_atomic_numbers())
        print()

    view(atoms_list)


def create_slab_population(
    atom_types: list[str] | list[int],
    cell_bounds: CustomCellBounds,
    population_size: int,
    closest_distances: CustomClosestDistances,
    number_of_atoms: int,
    volume: float = 9000.0,
    number_of_variable_cell_vectors: int = 0,
    splits: dict[tuple[int], int] = {(2,): 1, (1,): 1},
    slab: ase.Atoms | None = ase.Atoms('', pbc=True),
) -> list[ase.Atoms]:
    splits = {(2,): 1, (1,): 1}

    if slab is None:
        slab = ase.Atoms('', pbc=True)

    if isinstance(atom_types[0], str):
        atom_types = [atomic_numbers[atom] for atom in atom_types]
        print("Atom types: ", atom_types)

    n_atom_types = len(atom_types)
    n_atoms = number_of_atoms // n_atom_types
    block = []
    for i in range(n_atom_types):
        block.append((atom_types[i], n_atoms))

    sg = StartGenerator(
        slab=slab,
        blocks=block,
        blmin=closest_distances._ase_closest_distances,
        box_volume=volume,
        number_of_variable_cell_vectors=number_of_variable_cell_vectors,
        cellbounds=cell_bounds._ase_cellbounds,
        splits=splits,
    )

    print()
    print("Creating slab population...")
    crystals = []
    for i in tqdm(range(population_size)):
        crystal = sg.get_new_candidate()
        crystals.append(crystal)

    return crystals
