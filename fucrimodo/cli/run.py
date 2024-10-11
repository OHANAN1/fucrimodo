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
        self.verbose = args.verbose
        self.input_file = args.input_file

        if not os.path.exists(self.input_file):
            print("The Path to the input-file does not exist.")
            sys.exit(1)

        from fucrimodo.core.utils import soap_parser
        self.target_features, self.soap_obj = soap_parser.load_soap_features_from_file(
            self.input_file
        )

    def run(self):
        from configs.multi_stage_search.main import main
        main(
            target_features=self.target_features,
            soap_obj=self.soap_obj,
        )
