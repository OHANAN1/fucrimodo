from fucrimodo.customs.ga_stage.mutations import Mutation
from fucrimodo.customs.ga_stage import mutations as mut
from fucrimodo.core.utils.closest_distances_class import CustomClosestDistances
from fucrimodo.core.utils.cellbounds_custom import CustomCellBounds

def get_optimize_mutations(closest_distances: CustomClosestDistances):
    optimize_mutations = [
        mut.elem_mut.PermutationMutation(closest_distances=closest_distances),
        mut.pos_mut.RattleMutation(closest_distances=closest_distances, n_top=1, rattle_strength=1),
        mut.pos_mut.RattleMutation(closest_distances=closest_distances, n_top=1, rattle_strength=0.1),
        mut.pos_mut.RattleMutation(closest_distances=closest_distances, n_top=1, rattle_strength=0.01),
        mut.pos_mut.RattleMutation(closest_distances=closest_distances, n_top=1, rattle_strength=0.001),
        # mut.sym_mut.GetConventionalCellMutation(
        #     closest_distances=closest_distances,
        #     symmetry_tol=0.3
        # )
    ]

    multi_opti_mut = mut.multi_mut.MultipleMutations(
        mutations=optimize_mutations,
        number_of_mutations=2,
        random_order=True,
        closest_distances=closest_distances
    )

    all_opti_muts = optimize_mutations + [multi_opti_mut]
    return all_opti_muts

def get_enlarge_free_mutations(
    closest_distances: CustomClosestDistances,
    soap_species: list[str]
):
    perm_mut = mut.elem_mut.PermutationMutation(
        closest_distances=closest_distances,
    )
    rattle_mut = mut.pos_mut.RattleMutation(
        closest_distances=closest_distances,
    )
    add_mut = mut.elem_mut.AddAtomsMutation(
        possible_elements=soap_species,
        closest_distances=closest_distances
    )
    del_mut = mut.elem_mut.DeleteRandomAtomsMutation(
        closest_distances=closest_distances
    )
    replace_mut = mut.elem_mut.ReplaceAtomsMutation(
        possible_elements=soap_species,
        closest_distances=closest_distances
    )
    soft_mut = mut.energy_mut.SoftMutation(
        closest_distances=closest_distances
    )
    min_tilt_mut = mut.cell_mut.MinimizeTiltMutation(
        closest_distances=closest_distances
    )
    # conv_cell_mut = mut.sym_mut.GetConventionalCellMutation(
    #     closest_distances=closest_distances,
    #     symmetry_tol=0.3
    # )
    rot_mut = mut.cell_mut.RotationMutation(
        closest_distances=closest_distances
    )
    enlarge_free_muts = [  # noqa
        perm_mut,
        rattle_mut,
        add_mut,
        del_mut,
        replace_mut,
        soft_mut,
        min_tilt_mut,
        # conv_cell_mut,
        rot_mut,
    ]
    enlarge_free_multi_mut = mut.multi_mut.MultipleMutations(
        mutations=enlarge_free_muts,
        number_of_mutations=2,
        random_order=True,
        closest_distances=closest_distances
    )
    return enlarge_free_muts, enlarge_free_multi_mut

def get_enlarge_mutations(
    closest_distances: CustomClosestDistances,
    cell_bounds: CustomCellBounds
):
    enlarge_mut = mut.cell_mut.EnlargeMutation(
        closest_distances=closest_distances,
        cell_bounds=cell_bounds
    )
    strain_mut = mut.cell_mut.StrainMutation(
        closest_distances=closest_distances,
        n_variable_cell_vectors=3,
        cell_bounds=cell_bounds,
    )
    cutout_mut = mut.cell_mut.CutoutMutation(
        closest_distances=closest_distances,
        cell_bounds=cell_bounds,
        tolerance=1.
    )
    return [enlarge_mut, strain_mut, cutout_mut]


def get_all_muts(
    closest_distances: CustomClosestDistances, 
    cell_bounds: CustomCellBounds,
    soap_species: list[str]
) -> list[Mutation]:

    enlarge_free_muts, enlarge_free_multi_mut = get_enlarge_free_mutations(
        closest_distances, soap_species
    )
    enlarge_muts = get_enlarge_mutations(
        closest_distances, cell_bounds
    )

    enlarge_multi_mut = mut.multi_mut.MultipleMutations(
        mutations= [enlarge_muts[0], enlarge_free_multi_mut],
        number_of_mutations=2,
        random_order=False,
        closest_distances=closest_distances
    )
    all_muts = enlarge_free_muts + enlarge_muts + \
        [enlarge_free_multi_mut, enlarge_multi_mut]

    return all_muts

