from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
from nbformat import NotebookNode, validate
import os

def get_run_info_string(run_info: dict) -> str:
    run_info_str = "## Run Info\n"
    for key, value in run_info.items():
        if key != "stage_info":
            run_info_str += f"- *{key}*: {value}\n \n"
    return run_info_str


def get_setup_cells(run_name: str) -> list[NotebookNode]:
    return [
        new_markdown_cell(f"# Run: {run_name}"),
        new_code_cell(
            "# Uncomment the following line to use interactive plots\n"
                "# %matplotlib widget \n \n"
                "from IPython.display import Markdown\n"
                "from fucrimodo.analysis.analyse_run import AnalyseRun\n\n"
                "run_analysis = AnalyseRun('.')\n"
        ),
        new_code_cell(
            "# Please select the statistic key that you want to analyze.\n"
                "print('Possible keys: ', run_analysis.get_shared_statistic_keys())"
        ),
        new_code_cell(
            "statistic_key = # <- Write desired key here"
        ),
]


def get_run_info_cells() -> list[NotebookNode]:
    return [
        new_markdown_cell("## Run Info"),
        new_code_cell("Markdown('Missing')")
    ]


def get_visualization_cells(target_crystal_path: str | None = None) -> list[NotebookNode]:
    visualization_cells = [
            new_markdown_cell("## Visualization"),
            new_markdown_cell("### Best Found Crystal"),
            new_code_cell(
                "from ase.visualize import view\n\n"
                "best_crystal_tuple = run_analysis.get_best_crystal_tuple(statistic_key)\n"
                "best_crystal = best_crystal_tuple[0]\n"
                "print(best_crystal)"
            ),
            new_code_cell(
                "view(best_crystal)"
            )
        ]
    if type(target_crystal_path) == str:
        file_name_tar = os.path.basename(target_crystal_path)
        visualization_cells += [
            new_markdown_cell("### Target"),
            new_code_cell(
                "from ase.io import read\n"
                    "import os\n"
                    f"target_crystal_path = os.path.join(run_analysis.run_results.run_dir, '{file_name_tar}')\n"
                    "target_crystal = read(target_crystal_path)\n"
            ),
            new_code_cell(
                "view(target_crystal)"
            )
        ]
    return visualization_cells


def get_run_statistics_cells() -> list[NotebookNode]:
    return [
        new_markdown_cell("## Run statistics"),
        new_code_cell(
            "# Look at doc string or documentation for customization.\n"
            "from fucrimodo.analysis.analyse_run import create_combined_statistics_development_plot\n"
            "create_combined_statistics_development_plot(\n"
            "    run_analysis,\n"
            "    statistics_key=statistic_key,\n"
            ")"
        )
    ]

