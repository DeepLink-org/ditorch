import glob
from prettytable import PrettyTable
import argparse
import xml.etree.ElementTree as XMLET
import json
import os


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
    test_result_files = glob.glob(test_result_dir + "/**/*.xml", recursive=True)
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
    print(table.get_string(fields=["file", "classname", "name", "status"]))
    summary_file_name = test_result_dir + "/summary_test_result.csv"
    with open(summary_file_name, "w") as f:
        f.write(table.get_csv_string())
    print(f"Summary {len(test_infos)} test results saved to {summary_file_name}")
    return test_infos, table


def write_test_info_to_json(test_infos, pytorch_test_result):  # noqa: C901
    passed_test_case = {}
    skipped_test_case = {}
    failed_test_case = {}
    for info in test_infos:
        if info['file'] not in passed_test_case:
            passed_test_case[info['file']] = []
        if info['file'] not in skipped_test_case:
            skipped_test_case[info['file']] = []
        if info['file'] not in failed_test_case:
            failed_test_case[info['file']] = []

        case_name = info["classname"] + "." + info["name"]

        if info["status"] == "passed":
            if case_name not in passed_test_case[info['file']]:
                passed_test_case[info["file"]].append(case_name)
        elif info["status"] == "skipped":
            if case_name not in skipped_test_case[info['file']]:
                skipped_test_case[info["file"]].append(case_name)
        elif info["status"] == "error":
            if case_name not in failed_test_case[info['file']]:
                failed_test_case[info["file"]].append(case_name)

    passed_case_file_name = pytorch_test_result + "/passed_test_case.json"
    skipped_case_file_name = pytorch_test_result + "/skipped_test_case.json"
    failed_case_file_name = pytorch_test_result + "/failed_test_case.json"
    never_device_tested_case_file_name = pytorch_test_result + "/never_tested_device_test_case.json"
    never_cpu_tested_case_file_name = pytorch_test_result + "/never_tested_cpu_test_case.json"
    with open(passed_case_file_name, "w") as f:
        f.write(json.dumps(passed_test_case))
    with open(skipped_case_file_name, "w") as f:
        f.write(json.dumps(skipped_test_case))
    with open(failed_case_file_name, "w") as f:
        f.write(json.dumps(failed_test_case))

    print(f"Summary {sum(len(v) for v in passed_test_case.values())} test results saved to {passed_case_file_name}")
    print(f"Summary {sum(len(v) for v in failed_test_case.values())} test results saved to {failed_case_file_name}")
    print(f"Summary {sum(len(v) for v in skipped_test_case.values())} test results saved to {skipped_case_file_name}")

    all_test_case_id_file = f"{pytorch_test_result}/test_case_ids/all_device_test_cases.json"
    if os.path.exists(all_test_case_id_file):
        all_test_case = {}
        with open(all_test_case_id_file, "r") as f:
            all_test_case.update(json.load(f))

        for info in test_infos:
            case_name = info["classname"] + "." + info["name"]
            if info['file'] in all_test_case.keys():
                if case_name in all_test_case[info['file']]:
                    all_test_case[info['file']].remove(case_name)
        with open(never_device_tested_case_file_name, "w") as f:
            f.write(json.dumps(all_test_case))

        print(f"Summary {sum(len(v) for v in all_test_case.values())} test results saved to {never_device_tested_case_file_name}")

    all_test_case_id_file = f"{pytorch_test_result}/test_case_ids/all_cpu_test_cases.json"
    if os.path.exists(all_test_case_id_file):
        all_test_case = {}
        with open(all_test_case_id_file, "r") as f:
            all_test_case.update(json.load(f))

        for info in test_infos:
            case_name = info["classname"] + "." + info["name"]
            if info['file'] in all_test_case.keys():
                if case_name in all_test_case[info['file']]:
                    all_test_case[info['file']].remove(case_name)
        with open(never_cpu_tested_case_file_name, "w") as f:
            f.write(json.dumps(all_test_case))

        print(f"Summary {sum(len(v) for v in all_test_case.values())} test results saved to {never_cpu_tested_case_file_name}")


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
    test_info, csv_table = summary_test_results(pytorch_test_result)
    write_test_info_to_json(test_info, pytorch_test_result)
