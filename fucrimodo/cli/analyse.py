import os
import sys

class CLICommand:
    """Analyse the data that was collected during a run."""

    @staticmethod
    def add_arguments(parser):
        add = parser.add_argument
        add('analysis_type', help='Possible values: notebook, run')
        add('run_dir', help='Directory where the run is located that should be analysed.')

        add('-v', '--verbose', action='store_true', help='More output.')
        add('-c', '--target_crystal_path', help = \
            'Path to the file where the target crystal is located. ' \
            'If given, the scripts will consider the target crystal in ' \
            'its analysis.' \
            '(file-type: all types accepted by ase.io.read. e.g. xyz, xsf).'
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
        self.analysis_type = args.analysis_type
        self.verbose = args.verbose

        if not os.path.exists(args.run_dir):
            print("Path does not exist")
            sys.exit(1)
        if args.run_dir[-1] == "/":
            self.run_dir = args.run_dir[:-1]
        else:
            self.run_dir = args.run_dir

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

    def run(self):
        print(f"Running Analyse script with arguments: ")
        print(f"\tanalysis_type: \t{self.analysis_type}")
        print(f"\trun_dir: \t{self.run_dir}")
        print(f"\tverbose: \t{self.verbose}")

        if self.analysis_type == "notebook":
            self.__notebook_gen()

    def __notebook_gen(self):
        """Methode to generate the results notebook."""
        import nbformat
        from nbformat.v4 import new_notebook
        from nbformat import validate
        from fucrimodo.analysis.analyse_run import AnalyseRun
        from configs.result_notebook_generators.default_cells_run_analysis \
            import get_setup_cells, get_run_info_cells, get_run_statistics_cells, get_visualization_cells

        run_dir = os.path.basename(self.run_dir)
        print("Creating notebook for run dir:", run_dir)

        analyse_run = AnalyseRun(self.run_dir)

        nb = new_notebook()
        nb.cells.extend(get_setup_cells(run_name=analyse_run.run_results.run_name))
        nb.cells.extend(get_visualization_cells(self.target_crystal_path))
        nb.cells.extend(get_run_info_cells())
        nb.cells.extend(get_run_statistics_cells())

        if self.target_crystal_path is not None:
            import shutil
            shutil.copyfile(
                self.target_crystal_path,
                os.path.join(
                    self.run_dir,
                    os.path.basename(self.target_crystal_path)
                )
            )

        try:
            validate(nb)
            print("The generated notebook is valid.")
        except nbformat.validator.NotebookValidationError as e:
            print(f"The notebook is invalid: {e}")

        # Speichern des Notebooks
        notebook_name = "results_notebook.ipynb"
        notebook_path = os.path.join(self.run_dir, notebook_name)
        with open(notebook_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        print(f"Notebook saved at: {notebook_path}")
        print("Done")
        print()
