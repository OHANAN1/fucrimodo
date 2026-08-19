import json
import os
import re

import click

from fucrimodo.customs.utils import get_n_atoms_from_additional_notes


def main(origin: str, out: str, n_atoms, verbose: bool) -> None:
    """Change number of atoms in target file.

    :param origin: the original target file
    :param out: the path where the new target file should be placed
    :param n_atoms: How the n_atoms parameter should be changed in the new target file.
    """
    try:
        n_atoms = int(n_atoms)  # type: ignore
    except Exception as e:
        raise click.ClickException(f"Could not convert n_atoms to int: {e}")

    assert type(n_atoms) is int and n_atoms >= 1, "N atoms not correctly entered."

    if os.path.isfile(out):
        raise click.ClickException("FileExistsError: The output file already exists.")

    assert os.path.isfile(origin), "No input file found."

    # change the new file
    with open(origin, "r") as f:
        data = json.load(f)
        additional_notes = data["additional_notes"]

    original_n_atoms = get_n_atoms_from_additional_notes(additional_notes)
    if n_atoms == original_n_atoms:
        click.ClickException("ValueError original file already has correct n_atoms.")

    # Copy the file
    import shutil

    shutil.copy(origin, out)

    # Use complicated regex:
    data["additional_notes"] = re.sub(
        pattern=r"(Number of atoms: )\d+",
        repl=rf"\g<1>{n_atoms}",
        string=additional_notes,
    )

    assert get_n_atoms_from_additional_notes(data["additional_notes"])

    with open(out, "w") as f:
        json.dump(data, f)
