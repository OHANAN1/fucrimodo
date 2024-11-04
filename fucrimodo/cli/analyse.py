import os
import sys
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
            '-s', '--save_dir',
            help= \
            'Save the results of the analysis in the provided directory. '
            'The results will not be displayed if a directory is provided. '
            'If no directory is provided, the results will only '
            'be displayed.'
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
        add(
            '-c',
            '--config',
            help=('Path to the configuration file. If no path is provided, '
                'the default configuration file will be used.'
                'For each analysis object, a different configuration file is '
                'used. Check out the defaults in the fucrimodo_lab to adjust '
                'the configuration to your needs.'
            )
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

        if args.save_dir is not None:
            if not os.path.exists(args.save_dir):
                print("Path does not exist")
                sys.exit(1)
            self.save_dir = args.save_dir
            self.show = False

        else:
            self.save_dir = None
            self.show = True

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

        if args.config is not None:
            from fucrimodo.utils.import_helper import import_from_path
            config = import_from_path(args.config, "custom_analysis_config")
            if not hasattr(config, "main"):
                print("The config file must contain a method called 'main'.")
                sys.exit(1)

            if not callable(config.main):
                print("The method 'main' in the config file must be callable.")
                sys.exit(1)

            self.config = config
        else:
            self.config = None

    def __analyse_run(self):
        # Use the provided configuration file if it exists
        if self.config is not None:
            main = self.config.main

        # If no configuration file is provided, use the default configuration
        else:
            from fucrimodo.lab_template.configs.analysis.run.default import main

        main(
            run_dir=self.dir_path,
            verbose=self.verbose,
            row=self.row,
            show=self.show,
            save_dir = self.save_dir
        )

    def __analyse_stage(self):
        # Use the provided configuration file if it exists
        if self.config is not None:
            main = self.config.main

        # If no configuration file is provided, use the default configuration
        else:
            from fucrimodo.lab_template.configs.analysis.stage.default import main

        main(
            stage_dir=self.dir_path,
            verbose=self.verbose,
            row=self.row,
            show=self.show,
            save_dir = self.save_dir
        )

    def __generate_notebook(self):
        if self.config is not None:
            main = self.config.main

        else:
            from fucrimodo.lab_template.configs.analysis.notebook.default import main

        main(
            run_dir=self.dir_path,
            notebook_name="results_notebook.ipynb",
            verbose=self.verbose
        )

    def run(self):
        print(f"Running Analyse script with arguments: ")
        print(f"\tanalysis_object: \t{self.analysis_type}")
        print(f"\tdir_path: \t{self.dir_path}")
        print(f"\tverbose: \t{self.verbose}")
        print()

        if self.analysis_object == "notebook":
            self.__generate_notebook()
        elif self.analysis_object == "run":
            self.__analyse_run()
        elif self.analysis_object == "stage":
            self.__analyse_stage()
        else:
            raise ValueError(
                "Provided analysis type not found, " 
                    "only 'notebook', 'run', 'stage' are allowed."
            )
