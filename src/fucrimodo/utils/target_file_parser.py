import json
from typing import Any
from fucrimodo.customs.global_soap_target import GlobalSOAP
import numpy as np
from pathlib import Path

type JSONKey = str | int | float | bool | None


def save_to_target_file(
    features: list | np.ndarray,
    descriptor_name: str,
    descriptor_parameters: dict,
    save_path: str | Path,
    additional_notes: str | dict[str, Any] | None = None,
) -> None:
    """
    Saves the SOAP feature vector as well as the parameters of the SOAP object
    in a json file.

    Given file-path must end with .json extension

    :param features: list of the calculated features
    :param descriptor_name: name of the descriptor used to calculate the features
    :param descriptor_parameters: parameters used to create the descriptor
    :param additional_notes: additional notes that should be saved to the file
        e.g. information about the atoms object used to calculate the features
    """
    path = Path(save_path)
    if path.suffix != ".json":
        raise ValueError("Given file path must end with a .json extension")

    if additional_notes is None:
        additional_notes = ""

    # Make sure features can be stored in json format
    if type(features) is np.ndarray:
        features = features.tolist()

    assert type(features) is list, "Please provide features in form of list or ndarray!"

    soap_features_input = {
        "additional_notes": additional_notes,
        "descriptor_name": descriptor_name,
        "parameters": descriptor_parameters,
        "features": features,
    }
    with open(save_path, "w") as f:
        json.dump(soap_features_input, f)


def load_target_file(save_path: str) -> tuple[GlobalSOAP, list, str]:
    """Load the information saved in a target file.

    The target file should be a json file with the following fields:
    {
        'descriptor_name': str,
            Name of the descriptor used to calculate the features

        'parameters': dict,
            Parameters used to create the descriptor

        'features': list,
            The feature vector that should be inverted

        'additional_notes': str, dict, Optional
            Additional notes that were saved to the file
    }

    :param save_path: path to the target file

    :return: tuple of the loaded descriptor, features and
        additional notes saved in the target file
    """
    # Load the json file
    with open(save_path, "r") as f:
        target_dict = json.load(f)

    # Check if all necessary fields are present
    assert "descriptor_name" in target_dict.keys(), (
        "The json input needs to have a field called 'descriptor_name'.",
        f"But keys are: {target_dict.keys()}",
    )
    assert isinstance(
        target_dict["descriptor_name"], str
    ), "The descriptor name needs to be a string."

    assert "parameters" in target_dict.keys(), (
        "The json input needs to have a field called 'parameters'.",
        f"But keys are: {target_dict.keys()}",
    )
    assert isinstance(
        target_dict["parameters"], dict
    ), "The descriptor parameters need to be a dictionary."

    assert "features" in target_dict.keys(), (
        "The json input needs to have a field called 'features'.",
        f"But keys are: {target_dict.keys()}",
    )
    assert isinstance(target_dict["features"], list), "The features need to be a list."

    if not "additional_notes" in target_dict.keys():
        print("No additional notes found in the target file.")
        target_dict["additional_notes"] = ""

    if target_dict["descriptor_name"] == "GlobalSOAP":
        descriptor_object = GlobalSOAP(**target_dict["parameters"])
    elif target_dict["descriptor_name"] == "CustomSOAP":
        # To allow loading files with old descriptor name also
        print("Using old API for GlobalSOAP: CustomSOAP")
        descriptor_object = GlobalSOAP(**target_dict["parameters"])
    else:
        raise NotImplementedError(
            f"Descriptor {target_dict['descriptor_name']} is not implemented"
        )

    return (
        descriptor_object,
        target_dict["features"],
        target_dict["additional_notes"],
    )
