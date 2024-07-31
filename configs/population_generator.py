from fucrimodo.customs import population_generator as crystal_creation
from fucrimodo.core.utils.cellbounds_custom import CustomCellBounds
import ase 

def get_start_pop_candidates(
        soap_species: list[str],
        cell_bounds: CustomCellBounds,
        population_size: int
    ) -> list[ase.Atoms]:

    start_pop_candidates = crystal_creation.create_one_atomic_crystals(
        atom_types=soap_species,
        cell_bounds=cell_bounds,
        total_number_of_atoms=population_size,
    )

    return start_pop_candidates
