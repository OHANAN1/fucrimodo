import os
import sys
from fucrimodo.analysis.analyse_run import AnalyseRun, AnalyseStage
from fucrimodo.analysis.results_class import StageResults

class CLICommand:
    """Analyse the data that was collected during a run."""

    @staticmethod
    def add_arguments(parser):
        add = parser.add_argument
        add('analysis_type', help='Possible values: notebook, run, stage')
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

    def run(self):
        print(f"Running Analyse script with arguments: ")
        print(f"\tanalysis_type: \t{self.analysis_type}")
        print(f"\trun_dir: \t{self.run_dir}")
        print(f"\tverbose: \t{self.verbose}")

        if self.analysis_type == "notebook":
            self.__notebook_gen()
        elif self.analysis_type == "run":
            self.__analyse_run()
        elif self.analysis_type == "stage":
            self.__analyse_stage()
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

        global_log = analyse_run.run_results.global_statistics_log

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

        while True:
            print("Please select the stage you want to analyse:")
            stage_id = input("Stage ID: ")

            path_to_stage_dir = os.path.join(self.run_dir, f"stage_{stage_id}")
            if not os.path.isdir(path_to_stage_dir):
                print("The stage directory does not exist.")
                continue
            else:
                break

        stage_results = StageResults(self.run_dir, int(stage_id))

        analyse_stage = AnalyseStage(
            stage_results=stage_results,
        )

        possible_stat_keys = analyse_stage.get_fitness_keys() + ["mutation", "crossover"]
        if self.statistics_key is None:
            self.statistics_key = self.__let_user_select_statistics_key(
                possible_stat_keys=possible_stat_keys
            )

        if self.statistics_key == "mutation":

            mut_log = analyse_stage.stage_results.mutation_log
            gen = mut_log.select("gen")

            print("Possible mutation keys:")
            for i, mut_hash in enumerate(mut_log.chapters.keys()):
                print(f"\t{i}: {mut_hash}")

            print()
            mut_hash_index = input("Please select the index you want to analyse: ")
            mut_hash = list(mut_log.chapters.keys())[int(mut_hash_index)]

            fig, ax = plt.subplots()
            for stat_type in ['called', 'failed', 'survivor']:
                mutation = mut_log.chapters[mut_hash].select(stat_type)
                ax.plot(gen, mutation, label=f"{stat_type}")

            ax.set_xlabel("Generation")
            ax.set_ylabel("mutation")
            ax.set_title(f"Mutation: {mut_hash}")

            plt.legend()
            plt.show()
        
        elif self.statistics_key == "crossover":
            gen = analyse_stage.stage_results.fitness_log.select("gen")

            cross_log = analyse_stage.stage_results.crossover_log

            fig, ax = plt.subplots()
            for cross_hash in cross_log.chapters.keys():
                for stat_type in ['called', 'failed', 'survivor']:
                    crossover = cross_log.chapters[cross_hash].select(stat_type)
                    ax.plot(gen, crossover, label=f"{cross_hash} {stat_type}")

            ax.set_xlabel("Generation")
            ax.set_ylabel("crossover")

            plt.legend()
            plt.show()

        else:
            gen = analyse_stage.stage_results.fitness_log.select("gen")

            fig, ax = plt.subplots()
            for fit_type in ["max", "min", "avg"]:
                fitness = analyse_stage.stage_results.fitness_log.chapters[self.statistics_key].select(fit_type)
                ax.plot(gen, fitness, label=f"{fit_type}")

            ax.set_xlabel("Generation")
            ax.set_ylabel(self.statistics_key)

            plt.legend()
            plt.show()
