from fucrimodo.customs.ga_stage import crossovers as cross

def get_exploration_crossovers(closest_distances, cell_bounds):
    return [
        cross.OnePointElementCrossover(closest_distances),
        cross.OnePointPositionCrossover(closest_distances),
        cross.UnitCellCrossover(closest_distances),
        cross.StackCellsCrossover(closest_distances, cell_bounds),
    ]
