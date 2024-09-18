class CLICommand:
    """Perform descriptor inversion on provided input file."""

    @staticmethod
    def add_arguments(parser):
        add = parser.add_argument
        add('input_file', help='Possible values: notebook, run')
        add('-v', '--verbose', action='store_true', help='More output.')

    @staticmethod
    def run(args):
        from ase.db.cli import main

        main(args)
