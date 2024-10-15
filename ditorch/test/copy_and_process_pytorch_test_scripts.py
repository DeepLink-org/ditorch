import argparse
import os
import shutil
import glob


def parase_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pytorch_dir",
        type=str,
        default=os.environ.get("TORCH_SOURCE_PATH", None),
        help="path to pytorch repo",
    )
    parser.add_argument(
        "--pytorch_test_temp",
        type=str,
        default="pytorch_test_temp",
        help="Copy the pytorch test script to this directory",
    )
    args = parser.parse_args()
    return args


def copy_add_ditorch_import_to_pytorch_test_file(pytorch_dir, pytorch_test_temp):
    shutil.copytree(pytorch_dir + "/test", pytorch_test_temp + "/test", dirs_exist_ok=True)
    shutil.copytree(pytorch_dir + "/tools", pytorch_test_temp + "/tools", dirs_exist_ok=True)
    # We only add a line "import ditorch" to the original pytorch test script, and make no changes other than that.
    tests_files = glob.glob(pytorch_test_temp + "/**/*.py", recursive=True)
    for file_path in tests_files:
        with open(file_path, "rt") as test_script:
            content = "import ditorch\n"
            content = test_script.read()
        index1 = content.find("import torch")
        index2 = content.find("from torch")
        index = max(index1, index2)
        if index1 >= 0:
            index = min(index, index1)
        if index2 >= 0:
            index = min(index, index2)
        if index >= 0:
            content = content[:index] + "import ditorch\n" + content[index:]
        else:
            continue
        with open(file_path, "w") as test_script:
            test_script.write(content)
        print(f'"import ditorch" has been added to the beginning of the {file_path} file line')

    print(f"All pytorch {len(tests_files)} test scripts have been processed")


if __name__ == "__main__":
    args = parase_args()
    copy_add_ditorch_import_to_pytorch_test_file(args.pytorch_dir, args.pytorch_test_temp)
