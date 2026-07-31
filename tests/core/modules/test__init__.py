def test_imports():
    from fucrimodo.core.modules import (
        FitnessFunction,
        PopulationGenerator,
        PopulationSelection,
        Stage,
        Individual,
        BreakCondition,
    )


def test_star_imports():
    import fucrimodo.core.modules

    assert hasattr(fucrimodo.core.modules, "__all__")
