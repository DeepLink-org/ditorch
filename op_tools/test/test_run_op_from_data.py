import os
import subprocess
import shutil


def find_files(base_dir, target_filename):
    matches = []
    for root, dirs, files in os.walk(base_dir):
        if target_filename in files:
            matches.append(os.path.join(root, target_filename))
    return matches


def run_command_in_sub_process(commands):
    print(commands)
    result = subprocess.run(commands, shell=True, text=True, capture_output=True)
    print(result.stdout)
    print(result.stderr)
    if result.returncode != 0:
        print(F"Test {commands} FAILED")
    else:
        print(F"Test {commands} PASSED")
    print("\n\n\n")


def run_op_from_data(file_path):
    file_dir = file_path[0:file_path.rfind("/")]
    if not os.path.exists(file_path) or not os.path.exists(file_dir + "/output.pth"):
        print(F"{file_path} not exists")
        return

    commands = f"python op_tools/run_op_from_data.py {file_dir} --sync_time_measure --run_times 2"
    run_command_in_sub_process(commands)

    commands = f"python op_tools/run_op_from_data.py {file_dir} --acc_check --run_times 1"
    run_command_in_sub_process(commands)


if __name__ == "__main__":
    shutil.copytree("op_capture_result", "op_capture_result_raw", dirs_exist_ok=True)

    found_files = find_files("op_capture_result_raw", "input.pth")

    for file_path in found_files:
        run_op_from_data(file_path)
