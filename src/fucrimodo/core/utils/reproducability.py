import shutil
import subprocess


def get_last_commit_msg(run_path: str) -> str:
    """Returns the last commit message of the specified git repo.

    This can be used to set the description of the :class:`MultiStageSearch`
    automatically to the last commit message.

    :run_path: Path to dir, where git command should be executed.
        Can be the git root or any place inside the desired git repo.
    """
    commit_msg = ""
    try:
        commit_msg = subprocess.check_output(
            ["git", "-C", f"{run_path}", "-P", "log", "-1", "--pretty=%B"],
        )
    except Exception as e:
        commit_msg = f"Could not get git status. Error {e}"
        print("Could not get git commit msg.")
    finally:
        return str(commit_msg)


def save_run_script(file_path, save_path):
    """Small util to consistently store run files.

    :file_path: Please use keyword __file__ in main script to store it.
    :save_path: Path to store the script.
    """
    shutil.copy(file_path, save_path)
