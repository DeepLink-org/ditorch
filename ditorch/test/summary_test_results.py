import glob
import json
from prettytable import PrettyTable


def summary_test_results(test_result_dir):
    test_result_files = glob.glob(test_result_dir + '/*/result_test_*.json')
    table = PrettyTable()
    table.field_names = ["test_case_id", "exit_code"]
    for file_name in test_result_files:
        try:
            with open(file_name, 'r') as f:
                test_result = json.load(f)
            table.add_row([test_result["command_id"], test_result["returncode"]])
        except Exception as e:
            print(f"Error reading file {file_name}: {e}")
    print(table)
    if table.rowcount > 0:
        summary_file = test_result_dir + '/summary_test_result.csv'
        with open(summary_file, 'w') as f:
            f.write(table.get_string())
        print(f"Summary test results saved to {summary_file}")


if __name__ == "__main__":
    summary_test_results('pytorch_test_result')
