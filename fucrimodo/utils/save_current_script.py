from fucrimodo.core.utils.data_handeling import RunData

def save_current_script(run_data: RunData) -> str:
    import os

    current_script = __file__
    target_file_path = os.path.join(
        run_data.run_dir, os.path.basename(current_script)
    )
    with open(current_script, 'r') as source_file:
        content = source_file.read()

    with open(target_file_path, 'w') as target_file:
        target_file.write(content)

    return target_file_path

