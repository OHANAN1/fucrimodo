import os
import sys


class CLICommand:
    """Utilities for fucrimodo."""

    @staticmethod
    def add_arguments(parser):
        add = parser.add_argument
        add(
            "utility_name",
            help="Give the name of the utility that should be run."
            'Options are: "create_target"\n'
            "create_target: Create a target file for the inversion. "
            "Either give an atoms file that is readable by ase.io.read (e.g. xyz)."
            "Or give the path to an ase database file here you also need to "
            "provide the id of the atoms object that should be used."
            "To select the parameters that should be used, you can provide a "
            "path to a config file with the -c flag."
            "The path is provided with the -p flag and the id with the -i flag."
            "Adds this path as a note to the target file."
            'This should be a python file that contains a method "main" that '
            "takes the following arguments: main(atoms: ase.Atoms) "
            "The method should return a tuple of four objects: \n"
            "- The target features as a list of floats \n"
            "- A kwargs dict of the parameters used to create the descriptor \n"
            '\tthis dict should also contain a key "descriptor" with the name of the descriptor \n'
            "- A string that gets added to the notes of the target file \n"
            "- A string with the name the target file should have. \n"
            "If not provided, the default config will be used.",
        )
        add("-v", "--verbose", action="store_true", help="More output.")
        add(
            "-s",
            "--save_dir",
            help=(
                "Directory where the utilitie should save the results. "
                "If not provided, it is set to the current working directory."
            ),
        )
        add(
            "-p",
            "--path",
            help=(
                "Path to the file or directory the utilitie should act on. "
                "If not provided but needed, it is set to the current working "
                "directory."
            ),
        )
        add(
            "-i",
            "--id",
            help=(
                "Id of the atoms object in the ase database file that should be "
                "used. Only if the utility needs an id."
            ),
        )
        add(
            "-c",
            "--config",
            help=(
                "Path to the config file. Must be a python file that "
                'contains a method "main". Depending on the utility it takes '
                "different arguments: \n"
                'For "create_target": main(ase.Atoms) -> '
                "tuple[descriptor_name: str, target_features: list, "
                "descriptor_kwargs: dict, notes: str, save_name: str].\n "
                "If not provided, the default config will be used."
            ),
        )

    @staticmethod
    def run(args):
        runner = Runner()
        runner.parse(args)
        runner.run()


class Runner:
    possible_utilities = ["create_target"]

    def __init__(self):
        self.args = None

    def parse(self, args):
        self.args = args
        self.verbose = args.verbose
        self.id = args.id
        self.utility_name = args.utility_name
        assert (
            self.utility_name in self.possible_utilities
        ), "The utility name is not valid. Possible utilities are: " + ", ".join(
            self.possible_utilities
        )

        if args.save_dir is not None:
            if not os.path.exists(args.save_dir):
                print("The Path to the save-dir does not exist.")
                sys.exit(1)
            self.save_dir = args.save_dir
        else:
            self.save_dir = os.getcwd()

        if args.path is not None:
            if not os.path.exists(args.path):
                print(
                    f"The Path '{args.path}' to the target file or directory does not exist."
                )
                sys.exit(1)
            self.path = args.path
        else:
            self.path = os.getcwd()

        if args.config is not None:
            from fucrimodo.utils.import_helper import import_from_path

            config = import_from_path(args.config, "custom_util_config")
            if not hasattr(config, "main"):
                print("The config file must contain a method called 'main'.")
                sys.exit(1)

            if not callable(config.main):
                print("The method 'main' in the config file must be callable.")
                sys.exit(1)

            self.config = config
        else:
            self.config = None

    def __create_target_file(self):
        # Check if all necessary arguments are provided and correct.
        assert (
            self.path is not None
        ), "The path to the atoms file or ase database must be provided."
        assert os.path.isfile(
            self.path
        ), f"The provided path '{self.path}' to the atoms file or ase database must be a file."

        # Load the atoms object.
        if self.path.endswith(".db"):
            assert (
                self.id is not None
            ), "The id of the atoms object in the ase database must be provided."
            from fucrimodo.utils import ase_tools

            database = ase_tools.connect_to_existing_database(self.path)
            atoms = database.get_atoms(id=self.id)

            add_to_notes = f"Path to the database: {self.path}\n"
            add_to_notes += f"Id of the atoms object: {self.id}\n"
        else:
            from ase.io import read

            atoms = read(self.path)
            add_to_notes = f"Path to the atoms file: {self.path}\n"

        # Check if the atoms object is an ase.Atoms object.
        import ase

        assert isinstance(
            atoms, ase.Atoms
        ), "The atoms object must be an ase.Atoms object."

        # If no config is provided, use the default one.
        if self.config is None:
            from fucrimodo.lab_template.configs.utils.create_target_file_data import (
                main,
            )
        else:
            main = self.config.main

        # Run the utility.
        (
            descriptor_name,
            features,
            descriptor_parameters,
            additional_notes,
            save_name,
        ) = main(atoms)
        additional_notes = add_to_notes + additional_notes

        from fucrimodo.utils.target_file_parser import save_to_target_file

        save_to_target_file(
            features,
            descriptor_name=descriptor_name,
            descriptor_parameters=descriptor_parameters,
            additional_notes=additional_notes,
            save_path=os.path.join(self.save_dir, save_name),
        )

    def run(self):
        """Run the utility."""
        # Based on the utility name, run the corresponding method
        if self.utility_name == "create_target":
            self.__create_target_file()
        else:
            raise NotImplementedError("The utility is not implemented yet.")
