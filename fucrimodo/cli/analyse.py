import os
import sys
from fucrimodo.analysis.analyse_run import AnalyseRun

class CLICommand:
    """Analyse the data that was collected during a run."""

    @staticmethod
    def add_arguments(parser):
        add = parser.add_argument
        add('analysis_type', help='Possible values: notebook, run')
        add(
            'run_dir',
            help=\
            "Directory where the results of the run where saved. " \
            "Should contain the files: crystals.db, run_info.json and" \
            "stage_NUM.json for each stage performed."
        )

        add('-v', '--verbose', action='store_true', help='More output.')
        add(
            '-c', '--target_crystal_path', 
            help = \
            'Path to the file where the target crystal is located. ' \
            'If given, the scripts will consider the target crystal in ' \
            'its analysis.' \
            '(file-type: all types accepted by ase.io.read. e.g. xyz, xsf).'
        )
        add(
            '-s', '--statistics_key', type=str,
            help=\
            'Key of the statistic that should be analyzed. ' \
            'If not provided, but necessary, the script will display all ' \
            'possible keys and will prompt the user select one.'
        )

    @staticmethod
    def run(args):
        runner = Runner()
        runner.parse(args)
        runner.run()


class Runner:
    def __init__(self):
        self.args = None
        self.calculator_name = None

    def parse(self, args):
        self.args = args
        self.analysis_type = args.analysis_type
        self.verbose = args.verbose
        self.statistics_key = args.statistics_key

        if not os.path.exists(args.run_dir):
            print("Path does not exist")
            sys.exit(1)
        if args.run_dir[-1] == "/":
            self.run_dir = args.run_dir[:-1]
        else:
            self.run_dir = args.run_dir

        if args.target_crystal_path is not None:
            from ase.io import read
            try:
                read(args.target_crystal_path)
            except Exception as e:
                raise ValueError(
                    "Could not load target crystal from provided path."
                    f"Error: {e}"
                )
        self.target_crystal_path = args.target_crystal_path

    def run(self):
        print(f"Running Analyse script with arguments: ")
        print(f"\tanalysis_type: \t{self.analysis_type}")
        print(f"\trun_dir: \t{self.run_dir}")
        print(f"\tverbose: \t{self.verbose}")

        if self.analysis_type == "notebook":
            self.__notebook_gen()
        elif self.analysis_type == "run":
            self.__analyse_run()
        else:
            raise ValueError("Provided analysis type not found")

    def __let_user_select_statistics_key(
        self, possible_stat_keys: list[str]
    ) -> str:
        """Prompts the user to select one of the possible keys

        :param possible_stat_keys: list of statistic keys that can be used to analyse the run(s).

        :returns: The selected statistics key.

        :raise AssertionError: If user input is not an integer 
            or if integer is to big.
        """

        print("_____________________________________________________")
        print("Please choose the statistics key you want to analyse.")
        print()
        for i, stat_key in enumerate(possible_stat_keys):
            print(f"\t{i}: {stat_key}")

        print()
        selected_index = input("Type one of the corresponding numbers on the left: ")
        assert type(selected_index) != int, "Please write an integer number"
        assert int(selected_index)+1 <= len(possible_stat_keys), "The number you selected is to big."

        statistics_key = possible_stat_keys[int(selected_index)]
        print(f" -> Selected Key: {statistics_key}")
        return statistics_key

    def __notebook_gen(self):
        """Methode to generate the results notebook."""
        import nbformat
        from nbformat.v4 import new_notebook
        from nbformat import validate
        from configs.result_notebook_generators.default_cells_run_analysis \
            import get_setup_cells, get_run_info_cells, get_run_statistics_cells, get_visualization_cells

        run_dir = os.path.basename(self.run_dir)
        print("Creating notebook for run dir:", run_dir)

        analyse_run = AnalyseRun(self.run_dir)

        nb = new_notebook()
        nb.cells.extend(get_setup_cells(run_name=analyse_run.run_results.run_name))
        nb.cells.extend(get_visualization_cells(self.target_crystal_path))
        nb.cells.extend(get_run_info_cells())
        nb.cells.extend(get_run_statistics_cells())

        if self.target_crystal_path is not None:
            import shutil
            shutil.copyfile(
                self.target_crystal_path,
                os.path.join(
                    self.run_dir,
                    os.path.basename(self.target_crystal_path)
                )
            )

        try:
            validate(nb)
            print("The generated notebook is valid.")
        except nbformat.validator.NotebookValidationError as e:
            print(f"The notebook is invalid: {e}")

        # Speichern des Notebooks
        notebook_name = "results_notebook.ipynb"
        notebook_path = os.path.join(self.run_dir, notebook_name)
        with open(notebook_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        print(f"Notebook saved at: {notebook_path}")
        print("Done")
        print()

    def __analyse_run(self):
        import matplotlib.pyplot as plt
        analyse_run = AnalyseRun(self.run_dir)

        if self.statistics_key is None:
            print()
            self.statistics_key = self.__let_user_select_statistics_key(
                possible_stat_keys=analyse_run.get_shared_statistic_keys()
            )
            print()

        analysis_dir=os.path.join(self.run_dir, "analysis_results")
        if not os.path.isdir(analysis_dir):
            os.mkdir(analysis_dir)

        from fucrimodo.analysis.analyse_run import create_combined_statistics_development_plot
        create_combined_statistics_development_plot(
            analyse_run,
            statistics_key=self.statistics_key,
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
        if self.target_crystal_path is not None:
            from ase.io import read
            import ase
            target_crystal = read(self.target_crystal_path)
            assert type(target_crystal) == ase.Atoms, "Provided crystal is not ase.Atoms object."
        else:
            target_crystal = None

        analysis_results_dict = analyse_run.get_analysis_results_dict(
            statistics_key=self.statistics_key, target_crystal=target_crystal
        )
        import pprint
        pprint.pprint(analysis_results_dict)
