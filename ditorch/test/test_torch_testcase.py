import os
import subprocess
import json
import shutil
from ditorch.test.command_runner import CommandRunner
from ditorch.test.summary_test_results import summary_test_results, get_tested_test_cases
import argparse


def copy_add_ditorch_import_to_pytorch_test_file(file_name, pytorch_dir, dest_dir):
    file_path = f"{pytorch_dir}/test/" + file_name
    os.makedirs(dest_dir, exist_ok=True)
    # We only add a line "import ditorch" to the original pytorch test script, and make no changes other than that.
    with open(file_path, "rt") as torch_test_source_script_file:
        content = "import ditorch\n"
        content += torch_test_source_script_file.read()
    new_file_name = dest_dir + "/" + file_name
    with open(new_file_name, "w") as new_file:
        new_file.write(content)
    print(f'"import ditorch" has been added to the beginning of the {new_file_name} file line')


def run_command_in_sub_process(commands):
    print(commands)
    result = subprocess.run(commands, shell=True, text=True, capture_output=True)
    print(result.stdout)
    print(result.stderr)
    if result.returncode != 0:
        print(f"Run {commands} FAILED")
    else:
        print(f"Run {commands} SUCCESS")
    print("\n\n\n")
    return result


def split_device_and_cpu_test_cases(test_case_ids):
    cpu_test_case_ids = {}
    device_test_case_ids = {}
    for test_script_file, test_cases in test_case_ids.items():
        for case in test_cases:
            if "cpu" in case or "CPU" in case:
                if test_script_file not in cpu_test_case_ids:
                    cpu_test_case_ids[test_script_file] = [case]
                else:
                    cpu_test_case_ids[test_script_file].append(case)
            else:
                if test_script_file not in device_test_case_ids:
                    device_test_case_ids[test_script_file] = [case]
                else:
                    device_test_case_ids[test_script_file].append(case)
    return device_test_case_ids, cpu_test_case_ids


def filter_tested_cases(test_case_ids, tested_test_cases):
    filtered_test_case_ids = {}
    tested_case_count = 0
    total_case_count = 0
    for test_script_file, test_cases in test_case_ids.items():
        total_case_count += len(test_cases)
        if test_script_file in tested_test_cases.keys():
            not_tested_cases = []
            for case in test_cases:
                if case not in tested_test_cases[test_script_file]:
                    not_tested_cases.append(case)
            if len(not_tested_cases) > 0:
                filtered_test_case_ids[test_script_file] = not_tested_cases
                tested_case_count += len(not_tested_cases)
        else:
            filtered_test_case_ids[test_script_file] = test_cases
            tested_case_count += len(test_cases)
    print(f"There are {tested_case_count} test cases after filtering, and there are {total_case_count} test cases before filtering.")
    return filtered_test_case_ids


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pytorch_dir",
        type=str,
        default=os.environ.get("TORCH_SOURCE_PATH", None),
        help="path to pytorch repo",
    )
    parser.add_argument(
        "--skip_discover_test_case",
        action="store_true",
        help="wether to skip discover test cases",
    )
    parser.add_argument("--max_workers", type=int, default=64, help="max workers number")
    parser.add_argument(
        "--pytorch_test_temp",
        type=str,
        default="pytorch_test_temp",
        help="Copy the pytorch test script to this directory",
    )
    parser.add_argument(
        "--pytorch_test_result",
        type=str,
        default="pytorch_test_result",
        help="The directory to save the result files generated using pytorch test case testing",
    )
    parser.add_argument(
        "--test_case_ids_file",
        type=str,
        default="all_test_cases.json",
        help="The name of the json file saved to run the test case, the file is in the {pytorch_test_result}/test_case_ids directory, such as test_unary_ufuncs.py.json",  # noqa: E501
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(args)
    pytorch_test_temp = args.pytorch_test_temp
    pytorch_test_result = args.pytorch_test_result
    pytorch_dir = args.pytorch_dir
    test_case_ids_file = args.test_case_ids_file

    if not args.skip_discover_test_case or not os.path.exists(f"{pytorch_test_result}/test_case_ids/all_test_cases.json"):
        run_command_in_sub_process(f"python ditorch/test/discover_pytorch_test_case.py --pytorch_dir {pytorch_dir} --pytorch_test_result {pytorch_test_result}")  # noqa: E501

    with open(f"{pytorch_test_result}/test_case_ids/{test_case_ids_file}", "r") as f:
        test_case_ids = json.load(f)

    tested_case = get_tested_test_cases(f"{pytorch_test_result}")
    test_case_ids = filter_tested_cases(test_case_ids, tested_case)

    if not os.path.exists(pytorch_test_temp):
        if not pytorch_dir:
            print("TORCH_SOURCE_PATH not set")
            return -1
        if not os.path.isdir(pytorch_dir):
            print(f"{pytorch_dir} is not exist")
            return -1
        print(f"TORCH_SOURCE_PATH: {pytorch_dir}")
        print(f"start copy pytorch source files to {pytorch_test_temp}")
        shutil.copytree(pytorch_dir + "/test", pytorch_test_temp + "/test", dirs_exist_ok=True)
        shutil.copytree(pytorch_dir + "/tools", pytorch_test_temp + "/tools", dirs_exist_ok=True)
        print(f"copy pytorch source files to {pytorch_test_temp} success")

        source_files = test_case_ids.keys()
        for file_name in source_files:
            copy_add_ditorch_import_to_pytorch_test_file(file_name, pytorch_dir, dest_dir=f"{pytorch_test_temp}/test")

    commands_list = []
    device_test_case_ids, cpu_test_cases_ids = split_device_and_cpu_test_cases(test_case_ids)

    for test_script_file, test_cases in device_test_case_ids.items():
        for case in test_cases:
            commands = f"python {test_script_file} {case} -v --save-xml {pytorch_test_result}/xml"
            commands_list.append(
                (
                    f"{test_script_file}.{case}",
                    commands,
                )
            )

    run_cpu_test_case_commands = f"python run_test.py -k cpu -v --save-xml {pytorch_test_result}/xml"
    commands_list.append(("all_cpu_test_cases", run_cpu_test_case_commands))

    testcase_runner = CommandRunner(commands_list, max_workers=args.max_workers, cwd=f"{pytorch_test_temp}/test")
    testcase_runner.run()
    summary_test_results(f"{pytorch_test_result}")


if __name__ == "__main__":
    main()
