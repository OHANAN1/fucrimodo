import os

import ase
from ase.build import bulk
from ase.io import read

from fucrimodo.core.utils.custom_soap import CustomSOAP
from fucrimodo.core.utils.soap_parser import save_to_soap_features_file


def main(input_path, output_file_path):

    # Load the input file and check if it is a valid ASE object
    target_crystal = read(input_path)

    assert type(target_crystal) == ase.Atoms, "Input file must be an ASE Atoms object."

    target_features_save_path = os.path.join(output_file_path)

    soap_obj = CustomSOAP(
        species=set(target_crystal.get_atomic_numbers().tolist()),
        r_cut=15.0,
        n_max=8,
        l_max=8,
        sigma=0.5,
        average="inner",
        periodic=True,
    )
    features = soap_obj.create(target_crystal)
    save_to_soap_features_file(
        soap_object=soap_obj, features=features, save_path=target_features_save_path
    )
    print(f"Saved target file at {target_features_save_path}.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create target file.")
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to input file. Must be readable by ASE.io.read.",
    )
    parser.add_argument(
        "output_path", type=str, help="Path to output file. Must be json-file."
    )
    args = parser.parse_args()
    main(args.input_path, args.output_path)
