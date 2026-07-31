# This comment is for testing
from fucrimodo.core.utils import reproducability
import os

from numpy.lib import save


def test_get_last_commig_msg():
    # Check if it can be executed
    reproducability.get_last_commit_msg(os.path.abspath(__file__))


def test_save_run_script(tmp_path):
    save_path = os.path.join(tmp_path, "file")
    reproducability.save_run_script(os.path.abspath(__file__), save_path)

    with open(save_path, "r") as f:
        assert f.read().startswith("# This comment is for testing")
