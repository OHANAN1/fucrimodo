import os
import sys

class CLICommand:
    """Perform descriptor inversion on provided input file."""

    @staticmethod
    def add_arguments(parser):
        add = parser.add_argument
        add(
            'input_file',
            help='Give the path to the file with the features that should ' \
                'be inverted and the parameters that define the descriptor. ' \
                '(file-type: json).'
        )
        add('-v', '--verbose', action='store_true', help='More output.')
        add(
            '-c', '--config', 
            help=('Path to the config file. Must be a python file that '
                'contains a method "main" that takes the following arguments: '
                'main(fucrimodo.core.multi_stage_search.MultiStageSearch). '
                'If not provided, the default config will be used.')
        )
        add(
            '-s', '--save_dir',
            help=('Directory where the dir should be created in which the '
                'results of the inversion will be saved. '
                'If not provided, the results will be saved in a dir created'
                'in the current working directory. ')
        )
        add(
            '-n', '--name',
            help=('Name of the run. This will be used as the name of the '
                'directory in which the results will be saved. '
                'If not provided, the name will be set to the current time '
                'and date.')
        )


    @staticmethod
    def run(args):
        runner = Runner()
        runner.parse(args)
        runner.run()


class Runner:
    def __init__(self):
        self.args = None

    def parse(self, args):
        self.args = args
        self.verbose = args.verbose
        self.input_file = args.input_file
        self.name = args.name

        if not os.path.exists(self.input_file):
            print("The Path to the input-file does not exist.")
            sys.exit(1)

        if args.save_dir is not None:
            if not os.path.exists(args.save_dir):
                print("The Path to the save-dir does not exist.")
                sys.exit(1)
            self.save_dir = args.save_dir
        else:
            self.save_dir = os.getcwd()

        from fucrimodo.core.utils import soap_parser
        self.target_features, self.soap_obj = soap_parser.load_soap_features_from_file(
            self.input_file
        )

        if args.config is not None:
            from fucrimodo.utils.import_helper import import_from_path
            run_config = import_from_path(args.config, "custom_run_config")
            if not hasattr(run_config, "main"):
                print("The config file must contain a method called 'main'.")
                sys.exit(1)

            if not callable(run_config.main):
                print("The method 'main' in the config file must be callable.")
                sys.exit(1)

            self.run_config = run_config
        else:
            # Import the default run config from the lab_template
            from importlib import import_module
            self.run_config = import_module("fucrimodo.lab_template.configs.run.benchmark_run")

    def run(self):
        """Run the inversion."""
        # Create the MultiStageSearch object
        from fucrimodo.core.multi_stage_search import MultiStageSearch
        multi_stage_search = MultiStageSearch(
            save_dir=self.save_dir,
            target_features=self.target_features,
            descriptor_object=self.soap_obj,
            descriptive_name=self.name,
        )

        self.run_config.main(multi_stage_search)
