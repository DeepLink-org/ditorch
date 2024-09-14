import os
import re
import ditorch


def split_script(input_file):
    """用于将测试脚本test_xxx.py拆分为import部分和代码主体部分"""
    with open(input_file, "r") as f:
        lines = f.readlines()
    import_lines = []
    for line in lines:
        if line.startswith("#"):
            import_lines.append(line)
            continue
        elif "=" in line or ":" in line:
            break
        else:
            import_lines.append(line)
    if import_lines:
        while import_lines[-1].startswith("#"):
            import_lines.pop()

    imports_text = "".join(import_lines)
    functions_text = "".join(lines[len(import_lines) :])
    return imports_text, functions_text


def process_src_code(src: str):
    device_torch = ditorch.framework.split(":")[0]
    import_patch = """import torch
import ditorch
"""
    imports_text, functions_text = split_script(src)

    # 对某些device不支持的符号进行mock
    # torch.cuda.get_device_capability()
    mock_code = ""
    if device_torch == "torch_npu":
        mock_code = """
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
"""
    custom_test_code = r"""
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
"""

    over_time_test_code = """
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
"""
    imports_text = (
        import_patch + mock_code + imports_text + custom_test_code + over_time_test_code
    )

    functions_text = re.sub(
        r"run\_tests\(\)",
        """
    unittest.main(testRunner=CustomTextTestRunner)
""",
        functions_text,
    )

    with open(src, "w", encoding="utf-8") as file:
        file.write(imports_text + functions_text)


if __name__ == "__main__":
    import argparse

    os.environ["DISPLAY_SKIPPED_TESTS"] = "True"
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="original pytorch test file")
    args = parser.parse_args()
    process_src_code(args.filename)
