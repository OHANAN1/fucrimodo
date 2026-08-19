import importlib.util
import os
import sys

import click


def import_from_path(file_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise FileNotFoundError(f"Could not find file: {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    loader = spec.loader
    if loader is None:
        raise FileNotFoundError(f"Could not load file: {file_path}")

    loader.exec_module(module)
    return module


class ConfigScript:
    """Thin wrapper around a user-provided or default config module. Please use with click"""

    def __init__(self, path: str):
        # Check that the path exists
        assert os.path.isfile(
            path
        ), f"Could not find the config at {path}. Did you give the correct path and are you in the fucrimodo_lab directory?"
        self.path = path

        # (`[..] fucli_config klingt witzig`~/mlb)
        module = import_from_path(path, "fucli_config")

        # Check that file at the path has a main() method
        if not hasattr(module, "main"):
            raise click.ClickException(
                "The config file must contain a method called 'main'."
            )
        if not callable(module.main):
            raise click.ClickException(
                "The method 'main' in the config file must be callable."
            )
        self._module = module

    def run(
        self,
        **params,
    ):
        self._module.main(**params)
