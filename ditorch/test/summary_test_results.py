import glob
import json
from prettytable import PrettyTable
import argparse
import xml.etree.ElementTree as XMLET


def load_test_results_from_xml(xml_file):
    tree = XMLET.parse(xml_file)
    root = tree.getroot()
    test_infos = []
    for testcase in root.iter("testcase"):
        attr_dict = testcase.attrib
        info = {
            "file": attr_dict["file"],
            "classname": attr_dict["classname"],
            "name": attr_dict["name"],
            "status": "passed",
            "info": "",
            "message": "",
            "line": attr_dict["line"],
        }

        error = testcase.find("error")
        failure = testcase.find("failure")
        skipped = testcase.find("skipped")
        if error is not None:
            info.update({"status": "error"})
            info.update({"info": error.attrib["type"]})
            info.update({"message": f"{error.attrib['message']}"})
        if failure is not None:
            info.update({"status": "failure"})
            info.update({"info": failure.attrib["type"]})
            info.update({"message": f"{failure.attrib['message']}"})
        if skipped is not None:
            info.update({"status": "skipped"})
            info.update({"info": skipped.attrib["type"]})
            info.update({"message": f"{skipped.attrib['message']}"})

        test_infos.append(info)
    return test_infos


def summary_test_results(test_result_dir):
    test_result_files = glob.glob(test_result_dir + "/xml/**/*.xml")
    test_infos = []
    for file_name in test_result_files:
        try:
            test_infos += load_test_results_from_xml(file_name)
        except Exception as e:
            print(f"Error reading file {file_name}: {e}")
    if len(test_infos) == 0:
        print("No test result found")
        return
    table = PrettyTable()
    table.field_names = test_infos[0].keys()
    for info in test_infos:
        table.add_row(info.values())
    print(table)
    summary_file_name = test_result_dir + "/summary_test_result.csv"
    with open(summary_file_name, "w") as f:
        f.write(table.get_csv_string())
    print(f"Summary {len(test_infos)} test results saved to {summary_file_name}")


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
