import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
from nbformat import NotebookNode, validate
import os
import json
import ase
from fucrimodo.analysis.analyse_run import AnalyseRun

from configs.result_notebook_generators.default_cells_run_analysis \
    import get_setup_cells, get_run_info_cells, get_run_statistics_cells, get_visualization_cells


def main(
    run_dir_path: str,
    analyse_run: AnalyseRun,
    target_crystal_path: str | None = None,
    ):
    if run_dir_path[-1] == "/":
        run_dir_path = run_dir_path[:-1]
    run_dir_name = os.path.basename(run_dir_path)
    print("Creating notebook for run dir:", run_dir_name)

    with open(f"{run_dir_path}/run_info.json", "r") as f:
        run_info = json.load(f)

    nb = new_notebook()
    nb.cells.extend(get_setup_cells(run_name=analyse_run.run_results.run_name))
    nb.cells.extend(get_visualization_cells(target_crystal_path))
    nb.cells.extend(get_run_info_cells())
    nb.cells.extend(get_run_statistics_cells())

    if target_crystal_path is not None:
        import shutil
        target_crystal_file = os.path.basename(target_crystal_path)
        shutil.copyfile(target_crystal_path, os.path.join(
            run_dir_path, target_crystal_file
        ))

    try:
        validate(nb)
        print("The generated notebook is valid.")
    except nbformat.validator.NotebookValidationError as e:
        print(f"The notebook is invalid: {e}")

    # Speichern des Notebooks
    notebook_name = "results_notebook.ipynb"
    notebook_path = os.path.join(run_dir_path, notebook_name)
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"Notebook saved at: {notebook_path}")
    print("Done")
    print()

if __name__ == "__main__":

    import os
    import sys
    from ase.io import read
    import argparse

    parser = argparse.ArgumentParser(description='Analyse Run Script')

    parser.add_argument(
        '-d', '--run_dir', type=str,
        help="Directory where the notebook should be saved. " \
    )
    parser.add_argument(
        '-c', '--target_crystal_path', type=str,
        help='(Optional) Path to the file where the target crystal is located. ' \
            'If given, the script will add additional cells to the notebook' \
            'and will copy the target crystal file to the run_dir.' \
            '(file-type: all files accepted by ase.io.read. e.g. xyz, xsf).'
    )

    args = parser.parse_args()
    if args.run_dir is None:
        print("Please give path to the run directory with flag -d (see help).")
        sys.exit()
    else:
        analyse_run = AnalyseRun(run_results=args.run_dir)

    if args.target_crystal_path is not None:
        try:
            target_crystal = read(args.target_crystal_path)
        except Exception as e:
            raise ValueError(
                f"Cannot load target crystal file. Error: {e}"
            )

        assert type(target_crystal) == ase.Atoms, \
            "Provided file is not an ase.Atoms object."
    else:
        args.target_crystal_path = None


    if not os.path.exists(args.run_dir):
        print("Path does not exist")
        sys.exit(1)

    main(args.run_dir, analyse_run, args.target_crystal_path)
