from .modules import Stage, Population
from .utils import data_handeling

class MultiStageSearch:
    def __init__(
        self,
        run_data: data_handeling.RunData,
    ) -> None:
        self.run_data = run_data
        self.current_stage_id = 0

    def run(
        self,
        population: Population,
        stage: Stage
    ) -> Population:
        self.current_stage_id += 1
        stage.id = self.current_stage_id

        print(f"Running stage {self.current_stage_id}: {stage.name}")

        population = stage.run(
            population=population,
            global_log=self.run_data.global_logbook,
            global_stats=self.run_data.global_statistics, 
        )

        stage.save_results(
            save_path = self.run_data.run_dir,
            crystals_db = self.run_data.crystal_database
        )

        return population
