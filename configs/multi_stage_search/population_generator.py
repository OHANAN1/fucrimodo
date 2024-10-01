from fucrimodo.core.modules import Population, Individual, population
from fucrimodo.customs import population_generator as crystal_creation
from fucrimodo.core.utils.cellbounds_custom import CustomCellBounds
import ase

def get_start_pop_candidates(
        soap_species: list[str],
        population_size: int
    ) -> Population:

    cell_bounds = CustomCellBounds({
        "a": [1, 4], "b": [1, 4], "c": [1, 4], 
        "alpha": [20, 160], "beta": [20, 160], "gamma": [20, 160]
    })

    start_pop_candidates = crystal_creation.create_one_atomic_crystals(
        atom_types=soap_species,
        cell_bounds=cell_bounds,
        total_number_of_atoms=population_size,
    )

    individual_list = []
    for atoms in start_pop_candidates:
        individual_list.append(
            Individual(
                symbols=atoms.get_chemical_symbols(),
                positions=atoms.get_positions(),
                cell=atoms.cell,
                pbc=atoms.pbc
            )
        )

    population = Population(individual_list)

    return population
