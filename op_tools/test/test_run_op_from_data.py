import os
import subprocess
import shutil


def run_command_in_sub_process(commands):
    print(commands)
    result = subprocess.run(commands, shell=True, text=True, capture_output=True)
    print(result.stdout)
    print(result.stderr)
    if result.returncode != 0:
        print(f"Test {commands} FAILED")
    else:
        print(f"Test {commands} PASSED")
    print("\n\n\n")


if __name__ == "__main__":
    data_file_dir = "op_capture_result_raw"
    shutil.copytree("op_capture_result", data_file_dir, dirs_exist_ok=True)

    commands = f"python op_tools/run_op_from_data.py {data_file_dir} --sync_time_measure --run_times 10"
    run_command_in_sub_process(commands)

    commands = f"python op_tools/run_op_from_data.py {data_file_dir} --sync_time_measure --run_times 1 --acc_check"
    run_command_in_sub_process(commands)

    shutil.rmtree("op_capture_result")
    shutil.move(data_file_dir, "op_capture_result")
