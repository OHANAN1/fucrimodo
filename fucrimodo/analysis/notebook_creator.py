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
    notebook_name: str = "results_notebook.ipynb"
):
    notebook = create_results_notebook(run_dir)

    try:
        validate(notebook)
        print("The generated notebook is valid.")
    except nbformat.validator.NotebookValidationError as e:
        print(f"The notebook is invalid: {e}")


    # Execute the notebook
    # Change the current working directory to the run directory to make sure
    # the notebook can access the data
    current_dir = os.getcwd()
    os.chdir(run_dir)
    client = NotebookClient(notebook)
    client.execute()

    # Zurück zum ursprünglichen Arbeitsverzeichnis
    os.chdir(current_dir)

    # Speichern des Notebooks
    notebook_path = os.path.join(run_dir, notebook_name)
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    print(f"Notebook saved at: {notebook_path}")
    print("Done")
    print()

    # Check how I can collaps cells:
    # with open(notebook_path, 'r') as f:
    #     notebook = nbformat.read(f, as_version=4)
    #
    # # Fold all chapters
    # # Füge collapsible Metadaten zu allen Markdown-Überschriften hinzu
    # for cell in notebook['cells']:
    #     cell.metadata.heading_collapsed = True
    #
    # with open(notebook_path, 'w') as f:
    #     nbformat.write(notebook, f)

