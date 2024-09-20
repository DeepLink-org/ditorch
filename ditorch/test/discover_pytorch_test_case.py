import unittest
import ditorch  # noqa: F401
import sys
import json


def discover_test_cases_recursively(suite_or_case):
    if isinstance(suite_or_case, unittest.TestCase):
        return [suite_or_case]
    rc = []
    for element in suite_or_case:
        rc.extend(discover_test_cases_recursively(element))
    return rc


def get_test_full_names(test_cases):
    full_name_list = []
    for case in test_cases:
        id = case.id()
        module_name = id.split(".")[0] + ".py"
        test_name = id[id.find(".") + 1 :]
        full_name = module_name + " " + test_name
        full_name_list.append(full_name)
    return full_name_list


def discover_all_test_case(path="."):
    loader = unittest.TestLoader()
    test_cases = loader.discover(path)
    test_cases = discover_test_cases_recursively(test_cases)
    return test_cases


def dump_all_test_case_id_to_file(test_cases, path="all_test_cases.json"):
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

    with open(path, "wt") as f:
        json.dump(test_case_ids, f)

    print(f"dumped {case_num} test cases from {len(test_case_ids)} files to {path}")
    return test_case_ids


if __name__ == "__main__":
    print(f"discover:{sys.argv}")
    path = sys.argv[1] if len(sys.argv) > 1 else "."
    all_tests_case = discover_all_test_case(path)
    dump_all_test_case_id_to_file(all_tests_case)
