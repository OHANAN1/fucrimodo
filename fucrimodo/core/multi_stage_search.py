from .modules import Stage, Population
from .utils import data_handeling
import os
import pickle

class MultiStageSearch:
    def __init__(
        self,
        run_data: data_handeling.RunData,
    ) -> None:
        self.run_data = run_data
        self.current_stage_id = 0

    def save_results(self):
        file_path = os.path.join(self.run_data.run_dir, "global_logbook.pickle")
        with open(file_path, "wb") as f:
            pickle.dump(self.run_data.global_logbook, f)

    def run(
        self,
        population: Population,
        stage: Stage
    ) -> Population:
        self.current_stage_id += 1
        stage.id = self.current_stage_id

        # Create a dict to store the results of the stage
        save_dir = os.path.join(
            self.run_data.run_dir, f"stage_{self.current_stage_id}"
        )
        os.mkdir(save_dir)

        print(f"Running stage {self.current_stage_id}:")
        print(f"Stage ID: {stage.id}")
        print(f"Poulation size: {population.size}")
        population = stage.run(
            population=population,
            global_log=self.run_data.global_logbook,
            global_stats=self.run_data.global_statistics, 
        )

        print(f"Saving results of stage {self.current_stage_id}: {stage.name}")
        stage.save_results(
            save_path = save_dir,
            crystals_db = self.run_data.crystal_database
        )

        return population
