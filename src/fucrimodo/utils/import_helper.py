import importlib.util
import sys

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
