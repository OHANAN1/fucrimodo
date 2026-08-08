def test_imports():
    from fucrimodo.core.abstracts import (
        FitnessFunction,
        PopulationGenerator,
        PopulationSelection,
        Stage,
        BreakCondition,
    )


def test_star_imports():
    import fucrimodo.core.abstracts

    assert hasattr(fucrimodo.core.abstracts, "__all__")
