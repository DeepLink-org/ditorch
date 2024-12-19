import os
import subprocess


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
    raw_data_dir = "op_tools_results"

    commands = f"python op_tools/run_op_from_data.py {raw_data_dir} --sync_time_measure --run_times 10"
    run_command_in_sub_process(commands)

    commands = f"python op_tools/run_op_from_data.py {raw_data_dir} --sync_time_measure --run_times 2 --acc_check"
    run_command_in_sub_process(commands)
