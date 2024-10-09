import os
import sys
# from fucrimodo.analysis.analysis_classes import AnalyseRun, AnalyseStage
# from fucrimodo.analysis.results_classes import StageResults
import pandas as pd

class CLICommand:
    """Analyse the data that was collected during a run."""

    @staticmethod
    def add_arguments(parser):
        add = parser.add_argument
        add('analysis_object', help='Possible values: notebook, run, stage')
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
            '-s', '--stage_id', type=str,
            help=\
            'ID of the stage that should be analyzed. ' \
            'If not provided, the user will be prompted to select a stage.' \
            'Only relevant if analysis_object is "stage".'
        )
        add(
            '-t', '--analysis_type', type=str,
            help=\
            'Type of analysis that should be performed. ' \
            'Depending on the analysis object, different types of analysis ' \
            'can be performed. ' \
            'For the analysis object "stage", the following types are possible: ' \
            'mutation, crossover, fitness'
        )
        add(
            '-i', '--index', type=int,
            help=\
            'Index of the item that should be analyzed. ' \
            'Depending on the analysis object and type, different items can be ' \
            'analyzed. ' \
            'E.g. for the analysis object "stage" and type "mutation", the index ' \
            'refers to the mutation operator that should be analyzed.'
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
        self.analysis_object = args.analysis_object
        self.analysis_type = args.analysis_type
        self.verbose = args.verbose
        self.stage_id = args.stage_id
        self.index = args.index

        if not os.path.exists(args.run_dir):
            print("Path does not exist")
            sys.exit(1)
        if args.run_dir[-1] == "/":
            run_dir = args.run_dir[:-1]
        else:
            run_dir = args.run_dir
        if os.path.exists(run_dir) and type(run_dir) == str:
            self.run_dir: str = run_dir
        else:
            raise ValueError("The provided run_dir is not a valid path.")

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

    def __let_user_select_key(
        self, 
        selector: list[str] | pd.DataFrame | str,
        header: str = "Please select one of the following keys:",
    ) -> int:
        """Prompts the user to select one of the items in the selection.

        :param header: The header that should be displayed. Something like
        :param selector: The object that contains the possible selections.
            If it is a list, the items will be displayed with an index next
            to them. If it is a pandas DataFrame, the string representation
            of the DataFrame will be displayed.
            If it is a string, the string will be displayed.

        :returns: The user selected index.

        :raise AssertionError: If user input is not an integer.
        """

        # Display the header
        print("_____________________________________________________")
        print(header)
        print()

        if type(selector) == list:
            for i, key in enumerate(selector):
                print(f"\t{i}: {key}")
        elif type(selector) == pd.DataFrame or type(selector) == str:
            print(selector)
        else:
            raise ValueError("Selector type not recognized.")

        print()
        selected_index = input("Selected Index (number on left): ")
        assert type(selected_index) != int, "Please write an integer number"

        return int(selected_index)

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

        global_log = analyse_run.run_results.global_log

        if self.statistics_key is None:
            print()
            self.statistics_key = self.__let_user_select_statistics_key(
                possible_stat_keys=analyse_run.get_global_statistics_keys()
            )
            print()

        analysis_dir=os.path.join(self.run_dir, "analysis_results")
        if not os.path.isdir(analysis_dir):
            os.mkdir(analysis_dir)

        # Clean this up and put it in analysis class
        gen = global_log.select("gen")
        gen = [i for i in range(len(gen))]
        stage_id = global_log.select("stage_id")

        fig, ax = plt.subplots()
        for fit_type in ["max", "min", "avg"]:
            fitness = global_log.chapters[self.statistics_key].select(fit_type)
            ax.plot(gen, fitness, label=f"{fit_type}")

        current_stage = 1
        for i, stage in enumerate(stage_id):
            if stage != current_stage:
                ax.axvline(x=i, color="black", linestyle="--", alpha=0.5)
                current_stage = stage

        ax.set_xlabel("Generation")
        ax.set_ylabel(self.statistics_key)

        plt.legend()
        plt.show()

        # Get the analysis results dict
        if self.target_crystal_path is not None:
            from ase.io import read
            import ase
            target_crystal = read(self.target_crystal_path)
            assert type(target_crystal) == ase.Atoms, "Provided crystal is not ase.Atoms object."
        else:
            target_crystal = None

        # analysis_results_dict = analyse_run.get_analysis_results_dict(
        #     statistics_key=self.statistics_key, target_crystal=target_crystal
        # )
        # import pprint
        # pprint.pprint(analysis_results_dict)

    def __analyse_stage(self):
        import matplotlib.pyplot as plt

        if self.stage_id is None:
            print("Please select the stage you want to analyse:")
            self.stage_id = int(input("Stage ID: "))
            print()

        stage_results = StageResults(self.run_dir, int(self.stage_id))
        analyse_stage = AnalyseStage(
            stage_results=stage_results,
        )

        if self.analysis_type is None:
            selected_index = self.__let_user_select_key(
                selector=analyse_stage.analysis_types,
                header="Please select the type of analysis you want to perform:"
            )
            self.analysis_type = analyse_stage.analysis_types[selected_index]

        if self.index is None:
            self.index = self.__let_user_select_key(
                selector=analyse_stage.get_overview_table(self.analysis_type),
                header=f"Please select the index of the {self.analysis_type} you want to analyse:"
            )

        fig, ax = plt.subplots()
        analyse_stage.perform_analysis(
            analysis_type=self.analysis_type, 
            row=self.index,
            ax=ax
        )
        plt.legend()
        plt.show()

    def run(self):
        print(f"Running Analyse script with arguments: ")
        print(f"\tanalysis_object: \t{self.analysis_type}")
        print(f"\trun_dir: \t{self.run_dir}")
        print(f"\tverbose: \t{self.verbose}")

        if self.analysis_object == "notebook":
            self.__notebook_gen()
        elif self.analysis_object == "run":
            self.__analyse_run()
        elif self.analysis_object == "stage":
            self.__analyse_stage()
        else:
            raise ValueError("Provided analysis type not found")
