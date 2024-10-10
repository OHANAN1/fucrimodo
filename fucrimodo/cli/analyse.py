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

    def run(self):
        print(f"Running Analyse script with arguments: ")
        print(f"\tanalysis_object: \t{self.analysis_type}")
        print(f"\trun_dir: \t{self.run_dir}")
        print(f"\tverbose: \t{self.verbose}")

        if self.analysis_object == "notebook":
            from fucrimodo.analysis import notebook_creator as nc
            nc.cli_runner(self.run_dir)
        elif self.analysis_object == "run":
            from fucrimodo.analysis import run_analysis as ra
            ra.cli_runner(self.run_dir)
        elif self.analysis_object == "stage":
            from fucrimodo.analysis import stage_analysis as sa
            sa.cli_runner(self.run_dir)
        else:
            raise ValueError("Provided analysis type not found")
