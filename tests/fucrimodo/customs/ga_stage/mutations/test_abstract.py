import pytest

from fucrimodo.customs.ga_stage.mutations.abstract import Mutation


def test_incomplete_setup(closest_distances):
    class Incomplete(Mutation):
        pass

    with pytest.raises(TypeError):
        Incomplete(closest_distances)  # type: ignore
