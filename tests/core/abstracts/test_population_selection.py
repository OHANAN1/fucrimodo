import pytest
from fucrimodo.core.abstracts import PopulationSelection


def test_subclass_without_generate_cannot_instantiate():
    # does not implement any abstract
    class Incomplete(PopulationSelection):
        pass

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore
