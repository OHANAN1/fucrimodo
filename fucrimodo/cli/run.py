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
        import random
        from fucrimodo.core import multi_ga_search as multi_ga
        from fucrimodo.core.utils import data_handeling

        import numpy as np
        from icecream import ic
        import warnings

        # ╔══════════════════════════════════════════════════════════╗
        # ║                      Debug Settings                      ║
        # ╚══════════════════════════════════════════════════════════╝

        log_enable = True
        warnings.filterwarnings("ignore")
        ic.disable()

        random.seed(42)
        np.random.seed(42)

        verbose = 3
        soap_species: list["str"] = self.soap_obj.species

        # ── Stages ──────────────────────────────────────────────────────────────
        from configs.stage_list import get_stage_list
        stage_list = get_stage_list(
            soap_object=self.soap_obj,
            target_soap_features=self.target_features,
            soap_species=soap_species,
        )

        # ── Global Statistics ───────────────────────────────────────────────────
        from configs.global_statistics import get_global_statistics_dict
        global_stats_dict = get_global_statistics_dict(
            soap_object=self.soap_obj,
            target_soap_features=self.target_features,
        )

        # ── Start Population Candidates ─────────────────────────────────────────
        from configs.population_generator import get_start_pop_candidates
        start_pop_candidates = get_start_pop_candidates(
            soap_species=soap_species,
            population_size=20
        )

        run_data = data_handeling.RunData(
            save_dir="data/processed/results/",
            soap_object=self.soap_obj,
            log_enable=log_enable,
            save_n_best_crystals=10,
            global_statistics_dict=global_stats_dict,
        )

        # ── Perform Run ─────────────────────────────────────────────────────────
        run_data.add_run_settings(
            stage_data_list=stage_list,
            verbose=verbose
        )

        ga_search = multi_ga.MultiGenAlgSearch(
            run_data=run_data,
        )
        ga_search.run(
            start_pop_candidates=start_pop_candidates,
        )

        # save_current_script(run_data)
