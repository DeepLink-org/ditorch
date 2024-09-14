import json
import os

# true_error_tests = total_disabled_tests - test_separately_ok_tests

disabled_test_json = "/deeplink_afs/zhangqiu/pytorch_test/general_device_test/unsupported_test_cases/torch_npu_disabled_tests.json"
test_errors_log = "/deeplink_afs/zhangqiu/pytorch_test/general_device_test/custom_tools/test_error_separately.log"

total_disabled_tests = {}
with open(disabled_test_json, 'r') as f:
    total_disabled_tests = json.load(f)

with open(test_errors_log, 'r') as f:
    lines = f.readlines()
    for i in range(len(lines)):
        if lines[i] == "OK":
            test_class = lines[i+1].split(",")[0].split(":")[1].strip()
            test_method = lines[i+1].split(",")[1].split(":")[1].strip()
            if f"{test_method} (__main__.{test_class})" in total_disabled_tests.keys():
                total_disabled_tests.pop(f"{test_method} (__main__.{test_class})", None)

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'true_test_error.json')

out_file = open(file_path, "w")
json.dump(total_disabled_tests, out_file, indent=2)