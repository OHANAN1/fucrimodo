from fucrimodo.analysis.analyse_run import AnalyseRun
from fucrimodo.analysis.results_class import RunResults
from fucrimodo.analysis.analyse_run import create_combined_statistics_development_plot
import matplotlib.pyplot as plt
import ase

# ╔══════════════════════════════════════════════════════════╗
# ║                      Example Usage                       ║
# ╚══════════════════════════════════════════════════════════╝

def main(
    run_dir: str, 
    target_crystal: ase.Atoms | None, 
    statistics_key: str | None = None
):
    # ── Load the run ────────────────────────────────────────────────────────
    run_results = RunResults(
        run_dir=run_dir
    )

    # ── Create Analysis object ──────────────────────────────────────────────
    analyse_run = AnalyseRun(run_results=run_results)

    if statistics_key is None:
        # ── Ask User to give desired statistics key ─────────────────────────────
        print()
        print("Please choose the statistics key you want to analyse.")
        possible_stat_keys = analyse_run.get_shared_statistic_keys()
        for i, stat_key in enumerate(possible_stat_keys):
            print(f"{i}: {stat_key}")

        selected_index = input("Type one of the corresponding numbers on the left:")
        assert type(selected_index) != int, "Please write an integer number"
        assert int(selected_index)+1 <= len(possible_stat_keys), "The number you selected is to big."
        statistics_key = possible_stat_keys[int(selected_index)]

    # ── Initialize analysis_dir ─────────────────────────────────────────────
    analysis_dir=os.path.join(
        run_dir, "analysis_results"
    )
    if not os.path.isdir(analysis_dir):
        os.mkdir(analysis_dir)

    create_combined_statistics_development_plot(
        analyse_run,
        statistics_key=statistics_key,
        display_stage_id=True,
        stage_id_x_offset=0.85,
        stage_id_y_pos=1.15,
        statistics_name="Ref. Similarity",
        statistics_symbol="S$_\\text{r}$",
        save_fig=False,
        y_lim=(-0.1, 1.1),
        legend_params=dict(
            bbox_to_anchor=(0.4, 1.03), loc="lower center", fontsize=25
        )
    )
    plt.show()

    # Get the analysis results dict
    analysis_results_dict = analyse_run.get_analysis_results_dict(
        statistics_key=statistics_key, target_crystal=target_crystal
    )
    import pprint
    pprint.pprint(analysis_results_dict)

if __name__ == "__main__":

    import os
    import sys

    import argparse
    parser = argparse.ArgumentParser(description='Analyse Run Script')

    parser.add_argument(
        '-d', '--run_dir', type=str,
        help="Directory where the results of the run where saved. " \
            "Should contain the files: crystals.db, run_info.json and" \
            "stage_NUM.json for each stage performed."
    )
    parser.add_argument(
        '-c', '--target_crystal', type=str,
        help='(Optional) Path to the file where the target crystal is located. ' \
            'If given, the script will compare the best found crystal to this target crystal.' \
            '(file-type: all files accepted by ase.io.read. e.g. xyz, xsf).'
    )
    parser.add_argument(
        '-s', '--statistics_key', type=str,
        help='(Optional) Key of the statistic that should be analyzed. ' \
            'If not provided, the script will display all possible keys and ' \
            'will prompt the user select one.'
    )

    args = parser.parse_args()
    if args.run_dir is None:
        print("Please give path to the run directory with flag -d (see help).")
        sys.exit()

    if args.target_crystal is None:
        print("No target crystal is given.")
        target_crystal = None
    else:
        from ase.io import read
        target_crystal = read(args.target_crystal)
        assert type(target_crystal) == ase.Atoms, f"Loaded wrong target crystal file. Loaded: {target_crystal}"

    if not os.path.exists(args.run_dir):
        print("Path does not exist")
        sys.exit(1)

    main(args.run_dir, target_crystal, args.statistics_key)
