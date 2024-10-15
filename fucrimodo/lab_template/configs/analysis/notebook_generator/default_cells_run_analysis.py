from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
from nbformat import NotebookNode, validate
import os
from fucrimodo.analysis import run_analysis as ra
from fucrimodo.analysis import stage_analysis as sa

def get_setup_cells(run_data: ra.RunData) -> list[NotebookNode]:
   return [
        new_markdown_cell(f"# Run: {run_data.name}"),
        new_code_cell(
            "# Uncomment the following line to use interactive plots\n"
                "# %matplotlib widget\n\n" 
                "from IPython.display import Markdown, HTML\n"
                "from ase.visualize import view\n"
                "from fucrimodo.analysis import run_analysis as ra\n"
                "from fucrimodo.analysis import stage_analysis as sa\n\n"
                "run_data = ra.RunData('.')\n"
        ),
    ]


def get_run_info_cells(run_data: ra.RunData) -> list[NotebookNode]:
    # Add general run info to the notebook
    run_info_cells = [
        new_markdown_cell("## Run Info"),
        new_markdown_cell("### Overview"),
        new_markdown_cell(
            ra.get_run_overview(run_data).T.to_html(header=False)
        )
    ]

    # Add information about global statistics to the notebook
    global_stats_overview = ra.get_global_statistics_overview(run_data)
    run_info_cells += [
        new_markdown_cell("### Global Statistics"),
        new_markdown_cell("#### Overview"),
        # Load the dataframe so it can be sorted or filtered
        new_code_cell("ra.get_global_statistics_overview(run_data)"),
    ]


    # Create a plot for each of the global statistics
    for i in range(len(global_stats_overview.index)):
        run_info_cells += [
            new_markdown_cell(f"#### {global_stats_overview.at[i, "names"]}"),
            new_code_cell(f"ra.plot_global_statistics(run_data, row = {i})")
        ]

    # Show the best found crystal
    run_info_cells += [
        new_markdown_cell("### Best Found Crystal"),
        new_markdown_cell(
            "Change the `global_stats_row` parameter to specify which attribute "
                "of the crystal determines if it is the best. <br>"
                "Look at the Global Statistics Overview table to see "
                "the different options. (Row number is on the left)"
        ),
        new_code_cell("global_stats_row = 0"),
        new_markdown_cell("#### Overview"),
        new_code_cell(
            "Markdown(ra.get_best_crystal_overview(run_data, global_stats_row).T.to_html(header=False))"
        ),
        new_markdown_cell("#### Visualisation"),
        new_code_cell(
                "best_crystal, _, _ = ra.get_best_crystal_tuple("
                "run_data=run_data,"
                "global_statistics_row=global_stats_row"
                ")\n"
                "view(best_crystal, viewer='x3d')"
        ),
    ]

    return run_info_cells


def get_stage_info_cells(run_data: ra.RunData) -> list[NotebookNode]:
    stage_info_cells = [new_markdown_cell("## Stages")]

    # Get the stages data from the run data
    stages_dict = run_data.stages

    # Add information about each stage to the notebook
    stage_id = 1
    while True:
        # If the stage is not in the stages dictionary, all stages have been 
        # processed
        if stage_id not in stages_dict:
            break

        stage_data = stages_dict[stage_id]

        # Add general stage overview to the notebook
        stage_info_cells += [
            new_markdown_cell(f"### Stage {stage_id}: {stage_data.name}"),
            new_markdown_cell("#### Overview"),
            new_markdown_cell(
                sa.get_stage_overview(stage_data).T.to_html(header=False)
            )
        ]

        # Add information about the fitness functions used in the stage
        stage_info_cells += [
            new_markdown_cell("#### Fitness Functions"),
            # Load the dataframe so it can be sorted or filtered
            new_code_cell(
                f"df = sa.get_fitness_overview(run_data.stages[{stage_id}]) "
                "# Feel free to sort or filter the dataframe\n"
                "HTML(df.to_html())"
            ),
            new_code_cell(
                "# Adjust row to the fitness function you want to plot\n"
                f"sa.plot_fitness_statistics(run_data.stages[{stage_id}], row=0)"
            )
        ]

        # Add information about the mutation and crossover used in the stage
        for modification_type in ["Mutation", "Crossover"]:
            stage_info_cells += [
                new_markdown_cell(f"#### {modification_type}s"),
                new_code_cell(
                    "df = sa.get_modification_overview("
                        f"run_data.stages[{stage_id}], '{modification_type}') "
                        "# Feel free to sort or filter the dataframe\n"
                        "HTML(df.to_html())"
                ),
                new_code_cell(
                    "# Adjust row to the operator you want to plot\n"
                    f"sa.plot_modification_statistics(run_data.stages[{stage_id}], '{modification_type}', row=0)"
                )
            ]

        stage_id += 1

    return stage_info_cells
