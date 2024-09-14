import os
import json

disabled_test_json = "/deeplink_afs/zhangqiu/pytorch_test/general_device_test/unsupported_test_cases/torch_npu_disabled_tests.json"
with open(disabled_test_json, 'r+') as f:
    content = json.load(f)
    for key in content:
        test_case = key.split()[0]
        test_class = key.split()[1].strip("()").split(".")[1]
        test_error = content[key][0].split(":")[0]
        test_error_val = content[key][0].split(":")[1]
        if (test_error == "RuntimeError") and (test_error_val.startswith(" call")):
            os.system(f"python -m unittest test_nn.{test_class}.{test_case}")
    