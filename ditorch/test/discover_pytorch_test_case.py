import unittest
import ditorch  # noqa: F401
import sys
import json
import os


def discover_test_cases_recursively(suite_or_case):
    if isinstance(suite_or_case, unittest.TestCase):
        return [suite_or_case]
    rc = []
    for element in suite_or_case:
        rc.extend(discover_test_cases_recursively(element))
    return rc


def discover_all_test_case(path="."):
    loader = unittest.TestLoader()
    test_cases = loader.discover(path)
    test_cases = discover_test_cases_recursively(test_cases)
    return test_cases


def dump_all_test_case_id_to_file(test_cases, path):
    os.makedirs(path, exist_ok=True)
    test_case_ids = {}
    case_num = 0
    for case in test_cases:
        case_id = case.id()
        print(case_id)
        case_num += 1
        module_name = case_id.split(".")[0] + ".py"
        test_name = case_id[case_id.find(".") + 1 :]
        if not module_name.startswith("test_"):
            continue

        if module_name not in test_case_ids:
            test_case_ids[module_name] = [test_name]
        else:
            test_case_ids[module_name].append(test_name)

    total_test_case_file_name = path + "/all_test_cases.json"
    with open(total_test_case_file_name, "wt") as f:
        json.dump(test_case_ids, f)

    for module_name, test_names in test_case_ids.items():
        single_module_test_case_file_name = path + "/" + module_name + ".json"
        with open(single_module_test_case_file_name, "wt") as f:
            json.dump({module_name: test_names}, f)
        print(f"dumped {len(test_names)} test cases from {module_name} files to {single_module_test_case_file_name}")

    print(f"dumped {case_num} test cases from {len(test_case_ids)} files to {total_test_case_file_name}")
    return test_case_ids


if __name__ == "__main__":
    print(f"discover:{sys.argv}")
    test_script_path = sys.argv[1] if len(sys.argv) > 1 else "."
    output_path = sys.argv[2] if len(sys.argv) > 2 else "."
    all_tests_case = discover_all_test_case(test_script_path)
    dump_all_test_case_id_to_file(all_tests_case, output_path)
