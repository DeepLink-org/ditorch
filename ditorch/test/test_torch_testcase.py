import os
import subprocess
import json
import atexit
import shutil


def copy_add_ditorch_import_to_pytorch_test_file(file_name, pytorch_dir, dest_dir):
    file_path = f"{pytorch_dir}/test/" + file_name
    os.makedirs(dest_dir, exist_ok=True)
    # We only add a line "import ditorch" to the original pytorch test script, and make no changes other than that.
    with open(file_path, "rt") as torch_test_source_script_file:
        content = torch_test_source_script_file.read()
        content = "import ditorch\n" + content
    new_file_name = dest_dir + "/" + file_name
    with open(new_file_name, "w") as new_file:
        new_file.write(content)


def run_command_in_sub_process(commands):
    print(commands)
    result = subprocess.run(commands, shell=True, text=True, capture_output=True)
    print(result.stdout)
    print(result.stderr)
    if result.returncode != 0:
        print(f"Run {commands} FAILED")
    else:
        print(f"Run {commands} PASSED")
    print("\n\n\n")
    return result


test_record_list = []


def dump_test_record_to_file():
    with open("pytorch_test_results.json", "w") as f:
        print(test_record_list)
        json.dump(test_record_list, f)


def main():
    pytorch_dir = os.environ.get("TORCH_SOURCE_PATH")
    if not pytorch_dir:
        print("TORCH_SOURCE_PATH not set")
        return -1
    if not os.path.isdir(pytorch_dir):
        print(f"{pytorch_dir} is not exist")
        return -1
    pytorch_test_temp = "pytorch_test_temp"
    shutil.rmtree(pytorch_test_temp, ignore_errors=True)
    shutil.copytree(pytorch_dir + "/test", pytorch_test_temp)

    # Finding runnable test cases needs to be done in a process where the device is available.
    # The current process is a clean process that does not use device(import ditorch).
    run_command_in_sub_process(f"python ditorch/test/discover_pytorch_test_case.py {pytorch_test_temp}")
    with open("all_test_cases.json", "r") as f:
        test_case_ids = json.load(f)

    source_files = test_case_ids.keys()
    for file_name in source_files:
        copy_add_ditorch_import_to_pytorch_test_file(file_name, pytorch_dir, dest_dir="pytorch_test_temp")

    for test_script_file, test_cases in test_case_ids.items():
        for case in test_cases:
            commands = f"cd  pytorch_test_temp && python {test_script_file} {case} -v --save-xml pytorch_test_result_xml"
            print(f"Running {commands}")
            result = run_command_in_sub_process(commands)
            test_record = {
                "test_script_file": test_script_file,
                "test_case": case,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "commands": commands,
            }
            test_record_list.append(test_record)


if __name__ == "__main__":
    main()


atexit.register(dump_test_record_to_file)
