import json
from fucrimodo.core.utils.custom_soap import CustomSOAP
from numpy.typing import NDArray
import numpy as np


def save_to_soap_features_file(
    features: NDArray, 
    soap_object: CustomSOAP,
    save_path: str
) -> None:
    """
    Saves the SOAP feature vector as well as the parameters of the SOAP object
    in a json file.

    Given file-path must end with .json extension
    """
    assert save_path[-5:-1] != ".json", (
        "Given file path must end with file that has .json extension"
    )

    soap_features_input = {
        "parameters": soap_object.get_init_params(), 
        "features": features.tolist()
    }
    with open(save_path, "w") as f:
        json.dump(soap_features_input, f)


def load_soap_features_from_file(
    save_path: str
) -> tuple[NDArray[np.float64], CustomSOAP]:
    """
    Loads the feature vector and the CustomSOAP object it 
    was calculated with from a soap_features.json file.

    The json file must have following fields:
    {
        'parameters': dict, 
            Dict of parameters that can be used to 
            initialize the CustomSOAP object

        'features': list,
            The feature vector calculated with the 
            SOAP object.
    }
    """
    with open(save_path, "r") as f:
        soap_features = json.load(f)

    assert "parameters" in soap_features.keys(), (
        "The json input needs to have a field called 'parameters'.",
        f"But keys are: {soap_features.keys()}"
    )
    assert "features" in soap_features.keys(), (
        "The json input needs to have a field called 'features'.",
        f"But keys are: {soap_features.keys()}"
    )

    soap_object = CustomSOAP(
        **soap_features["parameters"]
    )
    features = soap_features["features"]
    features = np.array(features)

    return features, soap_object
