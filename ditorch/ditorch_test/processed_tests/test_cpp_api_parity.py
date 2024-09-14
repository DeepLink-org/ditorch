import torch
import ditorch

from unittest.mock import patch
patch('torch.cuda.get_device_capability', return_value=(8, 0)).start()

import torch_npu
if not hasattr(torch._C, '_cuda_setStream'):
    def _cuda_setStream(*args, **kwargs):
        pass
    setattr(torch._C, '_cuda_setStream', _cuda_setStream)
patch('torch._C._cuda_setStream', new=torch_npu._C._npu_setStream).start()

if not hasattr(torch._C, '_cuda_setDevice'):
    def _cuda_setDevice(*args, **kwargs):
        pass
    setattr(torch._C, '_cuda_setDevice', _cuda_setDevice)
patch('torch._C._cuda_setDevice', new=torch_npu._C._npu_setDevice).start()

if not hasattr(torch.backends.cudnn, 'is_acceptable'):
    def is_acceptable(*args, **kwargs):
        pass
    setattr(torch.backends.cudnn, 'is_acceptable', is_acceptable)
patch('torch.backends.cudnn.is_acceptable', return_value=True).start()

if not hasattr(torch.backends.cudnn, 'version'):
    def version(*args, **kwargs):
        pass
    setattr(torch.backends.cudnn, 'version', version)
patch('torch.backends.cudnn.version', return_value=90000).start()
# Owner(s): ["module: cpp"]

import torch
# NN tests use double as the default dtype
torch.set_default_dtype(torch.double)

import os

import torch.testing._internal.common_utils as common
import torch.testing._internal.common_nn as common_nn
from cpp_api_parity.parity_table_parser import parse_parity_tracker_table
from cpp_api_parity.utils import is_torch_nn_functional_test
from cpp_api_parity import module_impl_check, functional_impl_check, sample_module, sample_functional


import os
import json
import unittest
from datetime import datetime

class CustomTextTestResult(unittest.TextTestResult):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skipped_tests = []
        self.all_EF_infos = []

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.skipped_tests.append((test, reason))

    def addFailure(self, test, err):
        super().addFailure(test, err)

        self.all_EF_infos.append((test, err))

    def addError(self, test, err):
        super().addError(test, err)
        self.all_EF_infos.append((test, err))


    def printErrors(self):
        super().printErrors()

        if self.skipped_tests:
            self.stream.writeln(self.separator1)
            for test, reason in self.skipped_tests:
                self.stream.writeln(f"Skip {self.getDescription(test)}: {reason}")

        # 将异常信息转换为字符串
        if self.all_EF_infos:
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            device_torch = ditorch.framework.split(":")[0]
            test_failed_json = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + \
                f"/failed_tests_record/{device_torch}_{current_time}.json"
            if not os.path.exists(test_failed_json) or os.path.getsize(test_failed_json) == 0:
                with open(test_failed_json, 'w') as f:
                    json.dump({}, f)
            with open(test_failed_json, 'r+') as f:
                try:
                    # 如果文件中有内容，则加载内容
                    f.seek(0)
                    content = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    # 如果文件为空或解析失败，则初始化为空字典
                    content = {}
                # 更新内容
                for test, err in self.all_EF_infos:
                    exctype, value, tb = err
                    need_value = str(value).split("\n")[0]
                    # 如果测试还未存在于字典中，添加新内容
                    if str(test) not in content:
                        content[str(test)] = [f"{exctype.__name__}: {need_value}", ["linux"]]

                # 将文件指针移到开头，写入更新后的内容
                f.seek(0)
                json.dump(content, f, indent=4)
                f.truncate()  # 截断文件以确保文件末尾的多余数据被清除


class CustomTextTestRunner(unittest.TextTestRunner):
    def _makeResult(self):
        return CustomTextTestResult(self.stream, self.descriptions, self.verbosity)

import threading
import time

# 自定义的 OverTimeError
class OverTimeError(Exception):
    pass

# 这里保存原始的 run 方法
original_run = TestCase.run

# 强制终止线程的方法
def _stop_thread(thread):
    if not thread.is_alive():
        return
    import ctypes
    ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread.ident), ctypes.py_object(SystemExit))

# 定义一个带超时的运行方法
def run_with_timeout(self, result=None):
    print(f"Class: {self.__class__.__name__}, Method: {self._testMethodName}")
    timeout = getattr(self, 'timeout', 60)  # 获取每个测试类定义的超时时间，默认60秒

    # 获取主线程的 _tls (如果存在的话)
    if hasattr(self, "_tls"):
        main_tls = self._tls
    else:
        main_tls = None

    def thread_target():
        # 确保子线程有自己的 _tls 并初始化
        if main_tls is not None:
            self._tls = threading.local()  # 创建子线程的 _tls 对象
            # 初始化主线程中的属性
            if hasattr(main_tls, 'precision'):
                self._tls.precision = main_tls.precision
            else:
                self._tls.precision = TestCase._precision  # 或者设置一个默认值
            if hasattr(main_tls, 'rel_tol'):
                self._tls.rel_tol = main_tls.rel_tol
            else:
                self._tls.rel_tol = TestCase._rel_tol  # 或者设置一个默认值
        else:
            # 如果主线程没有 _tls，也在子线程中设置默认值
            self._tls = threading.local()
            self._tls.precision = TestCase._precision
            self._tls.rel_tol = TestCase._rel_tol

        # 运行原始的测试方法
        original_run(self, result)

    # 在单独的线程中运行测试
    test_thread = threading.Thread(target=thread_target)
    test_thread.daemon = True
    test_thread.start()
    test_thread.join(timeout)

    if test_thread.is_alive():
        # 测试超时，标记为失败
        result.addFailure(self, (OverTimeError, OverTimeError(f"Test exceeded time limit of {timeout} seconds."), None))
        # 强制停止超时线程
        _stop_thread(test_thread)

# 替换原始的 TestCase.run 方法
TestCase.run = run_with_timeout
# NOTE: turn this on if you want to print source code of all C++ tests (e.g. for debugging purpose)
PRINT_CPP_SOURCE = False

devices = ['cpu', 'cuda']

PARITY_TABLE_PATH = os.path.join(os.path.dirname(__file__), 'cpp_api_parity', 'parity-tracker.md')

parity_table = parse_parity_tracker_table(PARITY_TABLE_PATH)

class TestCppApiParity(common.TestCase):
    module_test_params_map = {}
    functional_test_params_map = {}

expected_test_params_dicts = []

if not common.IS_ARM64:
    for test_params_dicts, test_instance_class in [
        (sample_module.module_tests, common_nn.NewModuleTest),
        (sample_functional.functional_tests, common_nn.NewModuleTest),
        (common_nn.module_tests, common_nn.NewModuleTest),
        (common_nn.new_module_tests, common_nn.NewModuleTest),
        (common_nn.criterion_tests, common_nn.CriterionTest),
    ]:
        for test_params_dict in test_params_dicts:
            if test_params_dict.get('test_cpp_api_parity', True):
                if is_torch_nn_functional_test(test_params_dict):
                    functional_impl_check.write_test_to_test_class(
                        TestCppApiParity, test_params_dict, test_instance_class, parity_table, devices)
                else:
                    module_impl_check.write_test_to_test_class(
                        TestCppApiParity, test_params_dict, test_instance_class, parity_table, devices)
                expected_test_params_dicts.append(test_params_dict)

    # Assert that all NN module/functional test dicts appear in the parity test
    assert len([name for name in TestCppApiParity.__dict__ if 'test_torch_nn_' in name]) == \
        len(expected_test_params_dicts) * len(devices)

    # Assert that there exists auto-generated tests for `SampleModule` and `sample_functional`.
    # 4 == 2 (number of test dicts that are not skipped) * 2 (number of devices)
    assert len([name for name in TestCppApiParity.__dict__ if 'SampleModule' in name]) == 4
    # 4 == 2 (number of test dicts that are not skipped) * 2 (number of devices)
    assert len([name for name in TestCppApiParity.__dict__ if 'sample_functional' in name]) == 4

    module_impl_check.build_cpp_tests(TestCppApiParity, print_cpp_source=PRINT_CPP_SOURCE)
    functional_impl_check.build_cpp_tests(TestCppApiParity, print_cpp_source=PRINT_CPP_SOURCE)

if __name__ == "__main__":
    common.
    unittest.main(testRunner=CustomTextTestRunner)

