import os
import re
from collections import defaultdict
from io import StringIO

import ase
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except ImportError:
    raise ImportError("Please install seaborn.")

from fucrimodo.analysis.multi_run_analysis import (
    MultiRunData,
    get_all_global_statistics_overview,
    get_multi_run_overview,
)
from fucrimodo.analysis.run_analysis import RunData
from fucrimodo.utils.target_file_parser import load_target_file
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor

pal = sns.color_palette(
    "colorblind",
)
PALETTE = [pal[9], pal[8], pal[2]]


def extract_regex_from_additional_notes(save_path: str, regex: str) -> str | None:
    _, _, additional_notes = load_target_file(save_path)

    # Specify the pattern to search for with regex
    # The standard target file parser can add such patterns to the additional
    # notes
    match = re.search(regex, additional_notes)
    if match:
        # If there is a match, return the volume
        return str(match.group(1))
    else:
        raise ValueError(
            f"Could not find the pattern in the additional notes of {save_path}"
            f" with the regex pattern {regex}. \n"
            f"Additional notes: {additional_notes}"
        )


def test_if_structure_and_target_match(
    original_structure: ase.Atoms,
    found_structure: ase.Atoms,
):
    original_structure = AseAtomsAdaptor.get_structure(original_structure)  # type: ignore
    found_structure = AseAtomsAdaptor.get_structure(found_structure)  # type: ignore

    sm_loose = StructureMatcher(
        ltol=0.3,
        stol=0.5,
        angle_tol=10,
        primitive_cell=True,
        scale=True,
        attempt_supercell=True,
        allow_subset=True,
    )
    sm_strict = StructureMatcher(
        ltol=0.2,
        stol=0.3,
        angle_tol=5,
        primitive_cell=True,
        scale=True,
        attempt_supercell=True,
        allow_subset=True,
    )
    return (
        sm_strict.fit(original_structure, found_structure),
        sm_loose.fit(original_structure, found_structure),
    )


def get_target_structure_from_input_file(run_data: RunData):
    """Load the target structure from the additional notes of the input file.

    It is assumed that the target structure is stored in the additional notes
    as a CIF string. The CIF string is extracted from the additional notes
    using a regex pattern. The CIF string is then loaded into an ASE Atoms
    object.
    """
    cif_string = extract_regex_from_additional_notes(
        os.path.join(run_data.dir_path, "input_file.json"), r"CIF:(.*)"
    )
    assert cif_string is not None, "Could not find CIF string in additional notes."

    cif_string = cif_string.replace("NEWLINE", "\n")
    cif_string = cif_string.replace("QUOTATION_MARK", '"')
    with StringIO(cif_string) as f:
        from ase.io import read

        target_structure = read(f, format="cif")

    return target_structure


def get_target_structures(
    multi_run_data: MultiRunData,
):
    target_structures = []
    for run_data in multi_run_data.runs:
        target_structure = get_target_structure_from_input_file(run_data)
        target_structures.append(target_structure)

    return target_structures


def get_best_structures(
    multi_run_data: MultiRunData, key: str = "Reference_Similarity"
) -> tuple[list[ase.Atoms], list[float]]:
    values_at_key = []
    best_structures = []
    for run_data in multi_run_data.runs:
        structures = run_data.structures
        key_value_pairs = run_data.key_value_pairs

        index_with_best_structure = max(
            range(len(key_value_pairs)), key=lambda i: key_value_pairs[i][key]
        )

        best_structures.append(structures[index_with_best_structure])
        values_at_key.append(key_value_pairs[index_with_best_structure][key])

    return best_structures, values_at_key


def main(
    dir_path: str,
    verbose: bool = True,
    row: int | None = None,
    show: bool = True,
    save_dir: str | None = None,
):
    multi_run_data = MultiRunData(dir_path)

    overview = get_multi_run_overview(multi_run_data)
    global_stats_overview = get_all_global_statistics_overview(multi_run_data)

    # Split min and max similarity appart
    global_stats_overview[["min_similarity", "max_similarity"]] = (
        global_stats_overview["Reference_Similarity_min_max"]
        .str.split(", ", expand=True)
        .astype(float)
    )

    stats_df = pd.concat([overview, global_stats_overview], axis=1)

    print("________________________________________________________")
    print("Stats Dataframe")

    target_structures = get_target_structures(multi_run_data)

    stats_df["target_structures"] = target_structures

    # Set natoms as identifier, since each group of same crystals
    # has specific natoms
    stats_df["target_structures_natoms"] = [len(a) for a in target_structures]
    print(stats_df)

    # Assign, which runs match with the target
    best_structures, similarities = get_best_structures(
        multi_run_data=multi_run_data, key="Reference_Similarity"
    )

    match_dict = defaultdict(list)
    for i, (best_structure, target_structure) in enumerate(
        zip(best_structures, target_structures)
    ):
        target_structure = target_structures[i]
        best_structure = best_structures[i]
        match_strict, match_loose = test_if_structure_and_target_match(
            target_structure, best_structure
        )

        match_dict["value"].append(match_loose)
        match_dict["count"].append(len(target_structure) if match_loose else None)
        match_dict["match_type"].append("loose")
        match_dict["n_atoms"].append(len(target_structure))

        match_dict["value"].append(match_strict)
        match_dict["count"].append(len(target_structure) if match_strict else None)
        match_dict["match_type"].append("strict")
        match_dict["n_atoms"].append(len(target_structure))

        match_converged = global_stats_overview["max_similarity"][i] >= 0.9
        match_dict["value"].append(match_converged)
        match_dict["count"].append(len(target_structure) if match_converged else None)
        match_dict["match_type"].append("converged")
        match_dict["n_atoms"].append(len(target_structure))

    stats_df = pd.concat([stats_df, pd.DataFrame(match_dict)], axis=1)
    match_df = pd.DataFrame(match_dict)
    print(match_df)

    fig, ax = plt.subplots(sharey=True, tight_layout=True)

    axes = [ax]
    # axes[0].set_xlabel("Number of atoms")
    axes[0].set_xlabel("Target ID")
    axes[0].set_ylabel("Match rate")
    sns.barplot(
        data=match_df,
        ax=axes[0],
        y="value",
        x="n_atoms",
        hue="match_type",
        err_kws=None,
        errorbar=None,
        palette=PALETTE,
        edgecolor=".5",
        gap=0.05,
        legend=True,
    )
    sns.despine()
    axes[0].legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False)
    axes[0].set_ylim(0, 1.05)

    axes[0].set_axisbelow(True)
    axes[0].grid()

    print("Strict matches for n atoms:")
    for n_atoms in np.sort(np.unique(match_df["n_atoms"])):
        print(
            f"\t{n_atoms}: ",
            match_df.query(
                f"match_type == 'strict' and n_atoms == {n_atoms} and value == True"
            ).shape[0],
        )

    print("Loose matches for n atoms:")
    for n_atoms in np.sort(np.unique(match_df["n_atoms"])):
        print(
            f"\t{n_atoms}: ",
            match_df.query(
                f"match_type == 'loose' and n_atoms == {n_atoms} and value == True"
            ).shape[0],
        )

    if save_dir is not None:
        file_path = f"{save_dir}/match_rate_per_n_atoms_{row}.png"
        plt.savefig(file_path)
        plt.close()
        if verbose:
            click.echo(f"Stored file at {file_path}.")
    else:
        plt.show()
