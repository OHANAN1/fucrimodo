import os
import sys
import numpy as np

from fucrimodo.core.utils.custom_soap import CustomSOAP

class CLICommand:
    """Perform descriptor inversion on provided input file."""

    @staticmethod
    def add_arguments(parser):
        add = parser.add_argument
        add(
            'input_file',
            help=(
                'Give the path to the input file with the features that should '
                'be inverted and the parameters that define the descriptor. '
                'Can be created with scripts. Docs will be added soon.'
                'Or give the path to a directory where multiple input files ' 
                'are stored. These files will then be processed in parallel.'
            )
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
        add(
            '-p', '--parallel',
            help=('Only used if multiple input files are provided. '
                'Define the number of parallel processes to use. '
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

    def parse(self, args):
        self.args = args
        self.verbose = args.verbose
        self.input_file = args.input_file
        self.name = args.name

        if args.parallel is None:
            self.parallel = 1
        else:
            self.parallel = int(args.parallel)

        if not os.path.exists(self.input_file):
            print("The Path to the input-file does not exist.")
            sys.exit(1)

        if args.save_dir is not None:
            if not os.path.exists(args.save_dir):
                print("The Path to the save-dir does not exist. Creating it.")
                os.makedirs(args.save_dir)

            self.save_dir = args.save_dir
        else:
            self.save_dir = os.getcwd()

        # Get either the standard run config or the custom run config provided 
        # by the user
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


    def __copy_input_file(self, run_dir: str, input_file: str):
        """Copy the input file to the provided run directory. 

        The file will be renamed to 'input_file.json'.
        """
        import shutil
        save_path = os.path.join(run_dir, "input_file.json")
        shutil.copy(input_file, save_path)

    def __get_input_files_from_dir(self, input_dir: str) -> list[str]:
        """Get all files in the provided directory."""

        # Sort the files lexicographically to ensure that the order is always 
        # the same on every system and for every run
        sorted_file_names = sorted(os.listdir(input_dir))

        # Set the full paths to each file
        input_files = []
        for f in sorted_file_names:
            input_files.append(os.path.join(input_dir, f))

        return input_files

    def __get_features_and_soap_obj(
        self
    ) -> list[tuple[CustomSOAP, list, str]] | tuple[CustomSOAP, list, str]:
        from fucrimodo.utils import target_file_parser
        # If the input file is a directory, get all files in the directory
        if os.path.isdir(self.input_file):
            print("Processing multiple input files.")
            # Set the attribute to indicate that multiple files are being processed
            feature_soap_tuples = []
            input_files = self.__get_input_files_from_dir(self.input_file)
            for input_file in input_files:
                target_tuples = target_file_parser.load_target_file(
                    input_file
                )
                feature_soap_tuples.append(target_tuples)

            # Check if any files were found in the directory
            assert len(feature_soap_tuples) > 0, "No files in the directory."

            return feature_soap_tuples

        # If the input file is a single file, load the features from the file
        else:
            print("Processing a single input file.")
            target_tuple = target_file_parser.load_target_file(
                self.input_file
            )
            return target_tuple

    def __run_single_file(self, features_soap_tuple: tuple[CustomSOAP, list, str]):
        from fucrimodo.core.multi_stage_search import MultiStageSearch
        multi_stage_search = MultiStageSearch(
            save_dir=self.save_dir,
            target_features=np.array(features_soap_tuple[1]),
            descriptor_object=features_soap_tuple[0],
            descriptive_name=self.name,
        )
        additional_notes = features_soap_tuple[2]

        # Copy the input file to the created run directory
        self.__copy_input_file(multi_stage_search.run_dir, self.input_file)

        # Run the inversion with the provided run config or the default run config
        self.run_config.main(multi_stage_search)

    def __run_multiple_files(self, target_tuples: list[tuple[CustomSOAP, list, str]]):
        from fucrimodo.core.multi_stage_search import MultiStageSearch
        import multiprocessing

        # Create a MultiStageSearch object for each feature and SOAP object tuple
        run_id = 1
        multi_stage_searches = []
        for soap_obj, features, additional_notes in target_tuples:
            multi_stage_search = MultiStageSearch(
                save_dir=self.save_dir,
                target_features=np.array(features),
                descriptor_object=soap_obj,
                descriptive_name=f"{self.name}_id_{run_id}",
            )
            multi_stage_searches.append(multi_stage_search)
            run_id += 1


        # Get the input files for each MultiStageSearch object
        input_files = self.__get_input_files_from_dir(self.input_file)
        for i, multi_stage_search in enumerate(multi_stage_searches):
            # Copy the correct input file to the corresponding run directory
            self.__copy_input_file(multi_stage_search.run_dir, input_files[i])

        # Run the inversion for each MultiStageSearch object in sequence
        print(f"Running {len(multi_stage_searches)} inversion runs in parallel on {self.parallel} processes.")
        with multiprocessing.Pool(processes=self.parallel) as pool:
            pool.map(self.run_config.main, multi_stage_searches)

    def run(self):
        """Run the inversion."""
        # Get the features and SOAP object/s from the input file/s
        target_tuples = self.__get_features_and_soap_obj()

        if isinstance(target_tuples, list):
            self.__run_multiple_files(target_tuples)
        else:
            self.__run_single_file(target_tuples)

