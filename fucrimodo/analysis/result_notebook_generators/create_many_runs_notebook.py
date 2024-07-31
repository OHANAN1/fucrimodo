import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
import os
import json
import pandas as pd


def get_run_info_string(run_info: dict) -> str:
    run_info_str = "## Run Info\n"
    for key, value in run_info.items():
        if key != "stage_info":
            run_info_str += f"- *{key}*: {value}\n \n"
    return run_info_str


def main(run_dir_path: str):
    if run_dir_path[-1] == "/":
        run_dir_path = run_dir_path[:-1]
    run_dir_name = os.path.basename(run_dir_path)
    print("Creating notebook for many runs dir:", run_dir_name)

    with open(f"{run_dir_path}/run_info.json", "r") as f:
        run_info = json.load(f)

    nb = new_notebook()
    nb.cells.append(
        new_markdown_cell(f"# Results of Run: {run_dir_name}")
    )

    setup_string = "# Uncomment the following line to use interactive plots\n"
    setup_string += "# %matplotlib widget \n \n"
    setup_string += "from IPython.display import Markdown\n"
    setup_string += "import json\n"
    setup_string += "from src.utils import analyse_results as ar\n\n"
    setup_string += "run_info = ar.get_run_info_dict('.')\n"
    nb.cells.append(
        new_code_cell(setup_string)
    )

    nb.cells.append(
        new_markdown_cell("## Run Info"),
    )
    nb.cells.append(
        new_code_cell("Markdown(ar.get_run_info_string(run_info))")
    )

    nb.cells.append(
        new_markdown_cell("## Most Similar Crystals")
    )
    nb.cells.append(
        new_code_cell(
            "best_crystals_df = ar.get_best_crystals_dataframe(\n"
            "   number_of_crystals_to_return=10,\n"
            ")\n"
            "best_crystals_df"
        )
    )

    nb.cells.append(
        new_markdown_cell("## Stage Infos")
    )
    for stage_index, stage_info in enumerate(
        run_info["stage_info"].values()
    ):
        stage_id = stage_index + 1

        nb.cells.append(
            new_markdown_cell(f"### Stage {stage_id}")
        )
        nb.cells.append(
            new_code_cell(
                f"Markdown(ar.get_stage_info_string(run_info, stage_id={stage_id}, show_details=False))"
            )
        )
        nb.cells.append(
            new_code_cell(
                f"data = ar.create_stage_plots({stage_id}, '.', dark_mode=True)"
            )
        )

    nb.cells.append(
        new_markdown_cell("## Visualizations")
    )
    nb.cells.append(
        new_markdown_cell("Target Crystal")
    )
    nb.cells.append(
        new_code_cell(
            "ar.view_target_crystal()"
        )
    )
    nb.cells.append(
        new_markdown_cell("Created Crystals")
    )
    nb.cells.append(
        new_code_cell(
            "ar.view_best_crystals(\n"
            "   relax=False,\n"
            "   conventional_cell=False,\n"
            ")"
        )
    )

    # Speichern des Notebooks
    notebook_name = "results_notebook.ipynb"
    notebook_path = os.path.join(run_dir_path, notebook_name)
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"Notebook saved at: {notebook_path}")
    print("Done")
    print()


if "__main__" == __name__:
    import sys

    try:
        many_runs_dir_path = sys.argv[1]
    except IndexError:
        print("Please use as: python path/to/script.py path/to/many_runs_dir")
        sys.exit(1)

    if not os.path.exists(many_runs_dir_path):
        print("Path does not exist")
        sys.exit(1)

    main(many_runs_dir_path)
