import os
import sys
# from fucrimodo.analysis.analysis_classes import AnalyseRun, AnalyseStage
# from fucrimodo.analysis.results_classes import StageResults
import pandas as pd
import argparse


class CLICommand:
    """Analyse the data that was collected during a run."""
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        add = parser.add_argument
        add(
            'analysis_object', 
            help='Possible values: notebook, run, stage'
        )
        add(
            'dir_path',
            help= \
            "Directory where the results of the run or stage where saved. "
            "If the analysis object is 'notebook', provide the directory "
            "where the results of the run are saved. The notebook will be "
            "saved in the same directory."
        )
        add(
            '-v', '--verbose', 
            action='store_true', 
            help='More output.'
        )
        add(
            '-s', '--show',
            action='store_true',
            help= \
            'Show the results. If not given, the results will be saved to '
            'a file in the provided directory.'
        )
        add(
            '-t', '--analysis_type', 
            help=\
            'Type of analysis that should be performed. '
            'Depending on the analysis object, different types of analysis '
            'can be performed. '
            'For the analysis object "stage", the following types are possible: '
            'Mutation, Crossover, Fitness (Upper case is required). '
            'If no type is provided, a general overview of the stage will be '
            'displayed.'
        )
        add(
            '-r',
            '--row',
            help=\
            'For the different analysis types, different rows can be analyzed. '
            'If no row is provided, all rows will be analyzed.'
            'The row number is the index of the row in the table that will be '
            'displayed when the analysis is run.'
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
        self.analysis_object: str = args.analysis_object
        self.analysis_type: str = args.analysis_type

        assert type(args.verbose) == bool, "The flag --verbose must be a boolean"
        self.verbose: bool = args.verbose

        assert type(args.show) == bool, "The flag --show must be a boolean"
        self.show: bool = args.show

        # Check if the provided row can be converted to an integer
        if args.row is not None:
            try:
                args.row = int(args.row)
            except ValueError:
                print("The flag --row must be an integer or None")
                sys.exit(1)

        assert type(args.row) == int or args.row is None, "The flag --row must be an integer or None"
        self.row: int = args.row

        if args.dir_path is not None:
            if not os.path.exists(args.dir_path):
                print("Path does not exist")
                sys.exit(1)
            self.dir_path = args.dir_path
        else:
            print("No path provided")
            sys.exit(1)

    def run(self):
        print(f"Running Analyse script with arguments: ")
        print(f"\tanalysis_object: \t{self.analysis_type}")
        print(f"\tdir_path: \t{self.dir_path}")
        print(f"\tverbose: \t{self.verbose}")
        print()

        if self.analysis_object == "notebook":
            from fucrimodo.analysis import notebook_creator as nc
            nc.cli_runner(self.dir_path, verbose=self.verbose)
        elif self.analysis_object == "run":
            from fucrimodo.analysis import run_analysis as ra
            ra.cli_runner(
                self.dir_path,
                verbose=self.verbose,
                row=self.row,
                show=self.show,
            )
        elif self.analysis_object == "stage":
            from fucrimodo.analysis import stage_analysis as sa
            sa.cli_runner(
                stage_dir = self.dir_path,
                verbose = self.verbose, 
                row = self.row,
                show = self.show,
                analysis_type = self.analysis_type,
            )
        else:
            raise ValueError(
                "Provided analysis type not found, " 
                    "only 'notebook', 'run', 'stage' are allowed."
            )
