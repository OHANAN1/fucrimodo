from ase.build import bulk
from ase.io import write
from fucrimodo.core.utils.soap_parser import save_to_soap_features_file
from fucrimodo.core.utils.custom_soap import CustomSOAP
import os

# ── Initialize path to example data ─────────────────────────────────────
data_dir_path = os.path.abspath("data")
if not os.path.isdir(data_dir_path):
    os.mkdir(data_dir_path)

example_data_dir_path = os.path.join(data_dir_path, "example")
if not os.path.isdir(example_data_dir_path):
    os.mkdir(example_data_dir_path)


# ── Create example target crystal ───────────────────────────────────────
target_cry_save_path = os.path.join(
    example_data_dir_path, "example_target_crystal.xsf"
)

target_crystal = bulk('Cu', 'fcc', a=3.6, cubic=True)
write(target_cry_save_path, target_crystal)

print(f"Created and saved target crystal at {target_cry_save_path}.")


# ── Create example target features file ─────────────────────────────────
target_features_save_path = os.path.join(
    example_data_dir_path, "example_target_file.json"
)

soap_obj = CustomSOAP(
    species=["Cu"],
    r_cut=15.0,
    n_max=8,
    l_max=8,
    sigma=0.5,
    average="inner",
    periodic=True
)
features = soap_obj.create(target_crystal)
save_to_soap_features_file(
    soap_object=soap_obj,
    features=features,
    save_path=target_features_save_path
)

print(f"Saved example target file at {target_features_save_path}.")
