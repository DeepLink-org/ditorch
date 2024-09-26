import glob
import json
from prettytable import PrettyTable
import argparse


def summary_test_results(test_result_dir):
    test_result_files = glob.glob(test_result_dir + "/*/result_test_*.json")
    table = PrettyTable()
    table.field_names = ["test_case_id", "exit_code"]
    passed_case_count = 0
    failed_case_count = 0
    for file_name in test_result_files:
        try:
            with open(file_name, "r") as f:
                test_result = json.load(f)
            table.add_row([test_result["command_id"], test_result["returncode"]])
            if test_result["returncode"] == 0:
                passed_case_count += 1
            else:
                failed_case_count += 1
        except Exception as e:
            print(f"Error reading file {file_name}: {e}")
    print(table)
    if table.rowcount > 0:
        summary_file = test_result_dir + "/summary_test_result.csv"
        with open(summary_file, "w") as f:
            f.write(table.get_csv_string())
        total_case_count = passed_case_count + failed_case_count
        print(
            f"Summary test results saved to {summary_file}, total {total_case_count}, {passed_case_count} passed, {failed_case_count} failed"  # noqa: E501
        )


def get_tested_test_cases(test_result_dir):
    test_result_files = glob.glob(test_result_dir + "/*/result_test_*.json")
    test_cases = {}
    for file_name in test_result_files:
        try:
            with open(file_name, "r") as f:
                test_result = json.load(f)
            test_script_file, test_case_id = test_result["command_id"].split(".py.")
            test_script_file += ".py"
            if test_script_file not in test_cases:
                test_cases[test_script_file] = [test_case_id]
            else:
                test_cases[test_script_file].append(test_case_id)
        except Exception as e:
            print(f"Error reading file {file_name}: {e}")
    return test_cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pytorch_test_result",
        type=str,
        default="pytorch_test_result",
        help="path to test result dir",
    )
    args = parser.parse_args()
    pytorch_test_result = args.pytorch_test_result
    summary_test_results(pytorch_test_result)
    test_cases = get_tested_test_cases(pytorch_test_result)
    test_case_num = sum([len(cases) for cases in test_cases.values()])
    print(f"total tested {test_case_num} test cases from {len(test_cases)} test script files")
