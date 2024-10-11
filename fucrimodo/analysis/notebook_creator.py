import nbformat
from nbformat.v4 import new_notebook
from nbformat import validate
from configs.result_notebook_generators.default_cells_run_analysis \
    import get_setup_cells, get_run_info_cells, get_stage_info_cells
        
import os
from .run_analysis import RunData

from nbclient.client import NotebookClient

def create_results_notebook(
    run_dir
) -> nbformat.notebooknode.NotebookNode:
    """Methode to generate the results notebook."""

    print("Creating notebook for run dir:", run_dir)

    run_data = RunData(run_dir)

    nb = new_notebook()
    nb.cells.extend(get_setup_cells(run_data))
    nb.cells.extend(get_run_info_cells(run_data))
    nb.cells.extend(get_stage_info_cells(run_data))

    return nb


def cli_runner(
    run_dir: str,
    notebook_name: str = "results_notebook.ipynb",
    fold_chapters: bool = True,
    run_notebook: bool = True,
    verbose: bool = False,
):
    notebook = create_results_notebook(run_dir)

    # Add Metadata to the notebook to make the headings collapsible
    # To enable folding run: jupyter nbextension enable collapsible_headings/main
    # If necessary install required extensions
    if fold_chapters:
        for cell in notebook['cells']:
            # Only look for Heading cells above level 1
            if cell['cell_type'] == 'markdown':
                if cell['source'].startswith('##'):
                    # Add metadata to the cell
                    cell.metadata.heading_collapsed = True

    # Execute the notebook
    if run_notebook:
        # Save the current working directory to change back to it later
        current_dir = os.getcwd()

        # Change the current working directory to the run directory to make sure
        # the notebook can access the necessary data and run the code
        os.chdir(run_dir)
        client = NotebookClient(notebook)
        client.execute()

        # Go back to the original working directory
        os.chdir(current_dir)

    else:
        # If the notebook is not executed, check if it is valid instead
        try:
            validate(notebook)
            print("The generated notebook is valid.")

        except nbformat.validator.NotebookValidationError as e:
            print(f"The notebook is invalid: {e}")

    # Save the notebook
    notebook_path = os.path.join(run_dir, notebook_name)
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    print(f"Notebook saved at: {notebook_path}")
    print("Done")
    print()
