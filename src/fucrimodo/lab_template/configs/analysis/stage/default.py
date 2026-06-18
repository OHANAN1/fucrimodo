import os
import matplotlib.pyplot as plt
from fucrimodo.analysis.stage_analysis import (
    get_stage_overview,
    get_fitness_overview,
    get_modification_overview,
    plot_fitness_statistics,
    plot_modification_statistics,
    StageData
)

def main(
    stage_dir: str,
    row: int | None = None,
    analysis_type: str | None = None,
    save_dir: str | None = None,
    show: bool = False,
    verbose: bool = False,
):
    # Load the stage data
    stage_data = StageData(stage_dir)

    print(f"{analysis_type} Overview:")

    # Depending on the analysis type, set a different overview dataframe and
    # plot a different plot function
    if analysis_type is not None:
        # Set the row to 0 if it is None
        if row is None:
            print("Row: 0 (default)")
            row = 0
        else:
            print(f"Row: {row}")
        print()

        if analysis_type == "Fitness":
            # Get the fitness overview dataframe
            print("Fitness Overview:")
            print()
            print(get_fitness_overview(stage_data))
            plot_fitness_statistics(
                stage_data = stage_data,
                row = row
            )
            print()

        elif analysis_type == "Mutation" or analysis_type == "Crossover":
            print(f"{analysis_type} Overview:")
            print()
            print(get_modification_overview(stage_data, analysis_type))
            print()
            plot_modification_statistics(
                stage_data = stage_data,
                row = row,
                modification_type = analysis_type
            )

        else:
            raise ValueError(
                "The given analysis type is not valid for this stage. "
                "Possible are 'Fitness', 'Mutation' and 'Crossover'. "
                    "(Upper case is required)"
            )

        # Show the plot or save them to a file
        if show:
            plt.show()
        else:
            if save_dir is not None:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                plt.savefig(f"{save_dir}/{analysis_type}_{row}_overview.png")

            else:
                plt.savefig(f"{analysis_type}_{row}_overview.png")
                plt.close()


    else:
        print("No analysis type given. Displaying general overview.")
        print("Stage Overview:")
        print(get_stage_overview(stage_data).T)
        print()

        print("-------------------")
        print("Fitness Overview:")
        print(get_fitness_overview(stage_data))
        print()

        print("-------------------")
        print("Mutation Overview:")
        print(get_modification_overview(stage_data, "Mutation"))
        print()

        print("-------------------")
        print("Crossover Overview:")
        print(get_modification_overview(stage_data, "Crossover"))
        print()


