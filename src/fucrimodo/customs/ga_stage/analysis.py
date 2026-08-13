from ...analysis.utils import load_dict_from_file
import pandas as pd


def load_ga_stage_attributes(
    dir_path: str,
    info_dict: dict,
) -> dict:
    """Return a dictionary with attributes specific to the stage type.

    :return: Mapping with keys:

        * ``parent_selection``: name of the parent selection operator.
        * ``survivor_selection``: name of the survivor selection operator.
        * ``break_condition``: break condition used by the stage.
        * ``mutations``: mutation statistics DataFrame.
        * ``crossovers``: crossover statistics DataFrame.

        Mutation and crossover statistics are loaded from ``mutations.json``
        and ``crossovers.json``. The structure follows
        :attr:`fitness_statistics`, but each ``results`` entry is a
        :class:`pandas.DataFrame` with columns ``called``, ``failed``,
        ``survivor`` and ``gen``.


    :raises AssertionError: If the ``results`` entry in
        ``mutations.json`` or ``crossovers.json`` is not a list.
    """

    # Get the dict from the mutations.json file
    def get_mod_df(mod_type):
        mod_dict = load_dict_from_file(dir_path, f"{mod_type}.json")
        assert (
            type(mod_dict["results"]) == list
        ), f"The results entry in the {mod_type}.json file is not a list."

        # Load each of the results entries in a Dataframe
        for i in range(len(mod_dict["results"])):
            mod_dict["results"][i] = pd.DataFrame(mod_dict["results"][i])

        # Create the modification Dataframe
        return pd.DataFrame(mod_dict)

    mutations_df = get_mod_df("mutations")
    crossover_df = get_mod_df("crossovers")

    return {
        "parent_selection": str(info_dict["parent_selection"]),
        "survivor_selection": str(info_dict["survivor_selection"]),
        "break_condition": str(info_dict["break_condition"]),
        "mutations": mutations_df,
        "crossovers": crossover_df,
    }
