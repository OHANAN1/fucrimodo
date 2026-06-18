import os
import sys

class CLICommand:
    """Initialize a new fucrimodo lab to brew the potion that will revert
    descriptors to structures.

    This utility initializes a new fucrimodo lab. The lab is a directory
    structure that contains files and directories that are used to store
    the results of the inversion process. 
    To run fucrimodo, no lab is needed. However, it is recommended to use
    a lab to keep the results organized and to make it easier to create
    new runs and configure how fucrimodo should work.
    """
    @staticmethod
    def add_arguments(parser):
        add = parser.add_argument
        add('-v', '--verbose', action='store_true', help='More output.')
        add(
            '-s', '--save_dir',
            help=('Directory where the lab should be created'
                'If not provided, it is set to the current working directory.')
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

        if args.save_dir is not None:
            if not os.path.exists(args.save_dir):
                print("The Path to the save-dir does not exist.")
                sys.exit(1)
            self.save_dir = args.save_dir
        else:
            self.save_dir = os.getcwd()

    def run(self):
        """Copy the fucrimodo_lab template to the desired location."""
        # User needs to confirm that the lab should be created at the given 
        # location.
        confirmation = input(f"Create fucrimodo lab in {self.save_dir})[y/n]: ")
        if confirmation.lower() != 'y':
            print("Aborted.")
            sys.exit(0)

        fucrimodo_lab_dir = os.path.join(self.save_dir, 'fucrimodo_lab')
        # Check if the directory already exists.
        if os.path.exists(fucrimodo_lab_dir):
            print(f"The directory {fucrimodo_lab_dir} already exists.")
            sys.exit(1)

        # Get the location of the lab template.
        import importlib.resources as pkg_resources
        lab_template = str(pkg_resources.files('fucrimodo').joinpath('lab_template'))

        # Copy the lab template to the desired location.
        import shutil
        shutil.copytree(lab_template, fucrimodo_lab_dir)

        # Print a message to the user.
        print(f"Lab created in {fucrimodo_lab_dir}.")
        print("You can configure the lab by editing the config files in the lab.")
        print("For more information, see the documentation.")
        print("Brew the potion that reverts the descriptor to its atomic form!")

