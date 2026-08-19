def test_imports():
    from fucrimodo.core.abstracts import (
        BreakCondition,
        FitnessFunction,
        PopulationGenerator,
        PopulationSelection,
        Stage,
    )


def test_star_imports():
    import fucrimodo.core.abstracts

    assert hasattr(fucrimodo.core.abstracts, "__all__")
