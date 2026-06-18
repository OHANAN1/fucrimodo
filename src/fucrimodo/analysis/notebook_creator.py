import nbformat
from nbformat.v4 import new_notebook
from nbformat import validate
import os
from nbclient.client import NotebookClient


def create_and_test_results_notebook(
    run_dir: str,
    cell_list: list[nbformat.notebooknode.NotebookNode],
    fold_chapters: bool = True,
    run_notebook: bool = True,
    verbose: bool = False,
    notebook_name: str = "results_notebook.ipynb",
):
    """Methode to generate the results notebook.

    :param run_dir: The directory of the run to create the notebook for.
        Here the notebook will be saved.
    :param fold_chapters: If True, the notebook will have collapsible headings.
    :param run_notebook: If True, the notebook will be executed. If False, the
        notebook will be validated instead.
    :param verbose: If True, additional information will be printed.
    """
    notebook = new_notebook()
    notebook.cells.extend(cell_list)

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
