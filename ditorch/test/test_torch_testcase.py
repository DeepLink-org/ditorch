import os


def run_test(file_path):
    os.makedirs("temp", exist_ok=True)
    with open(file_path, "rt") as torch_test_source_script_file:
        content = torch_test_source_script_file.read()
        content = "import ditorch\n" + content
    new_file_name = "temp/" + file_path.split("/")[-1]
    pass
    with open(new_file_name, "w") as new_file:
        new_file.write(content)
    print(f"test {file_path} over")


TORCH_TEST_SCRIPT_FILE = [
    "test_ops.py",
]


def main():
    pytorch_dir = os.environ.get("PYTORCH_SOURCE_DIR")
    if not pytorch_dir:
        print("PYTORCH_SOURCE_DIR not set")
        return -1
    if not os.path.isdir(pytorch_dir):
        print(f"{pytorch_dir} is not exist")
        return -1
    for file_path in TORCH_TEST_SCRIPT_FILE:
        full_path = pytorch_dir + "/test/" + file_path
        run_test(full_path)


if __name__ == "__main__":
    main()
