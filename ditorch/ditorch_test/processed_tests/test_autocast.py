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
# Owner(s): ["module: unknown"]

import collections
import unittest

import torch
from torch.testing._internal.common_utils import TestCase, run_tests, IS_WINDOWS
from torch.testing._internal.autocast_test_lists import AutocastCPUTestLists
from torch.utils._python_dispatch import TorchDispatchMode


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
class TestAutocastCPU(TestCase):
    def setUp(self):
        super().setUp()
        self.autocast_lists = AutocastCPUTestLists(torch.device('cpu'))

    def tearDown(self):
        del self.autocast_lists
        super().tearDown()

    def _run_autocast_outofplace(self, op, args, run_as_type, out_type=None, module=torch, add_kwargs=None):
        # helper to cast args
        def cast(val, to_type):
            if isinstance(val, torch.Tensor):
                return val.to(to_type) if val.is_floating_point() else val
            elif isinstance(val, collections.abc.Iterable):
                return type(val)(cast(v, to_type) for v in val)
            else:
                return val

        if add_kwargs is None:
            add_kwargs = {}

        self.assertFalse(torch.is_autocast_cpu_enabled())
        with torch.cpu.amp.autocast():
            self.assertTrue(torch.is_autocast_cpu_enabled())
            out_type = out_type if out_type is not None else run_as_type
            output = output_method = None

            # Try module.* variant, if requested:
            if module is not None and hasattr(module, op):
                output = getattr(module, op)(*args, **add_kwargs)
                if isinstance(output, torch.Tensor):
                    self.assertTrue(out_type == output.dtype,
                                    f"autocast for torch.{op} produced {output.dtype}, should produce {out_type}")
            # Try Tensor.* variant:
            if hasattr(torch.Tensor, op):
                output_method = getattr(args[0], op)(*args[1:], **add_kwargs)
                if isinstance(output_method, torch.Tensor):
                    self.assertTrue(out_type == output_method.dtype,
                                    "autocast for torch.{} produced {}, should produce torch.{}"
                                    .format(op, output_method.dtype, out_type))

            self.assertTrue((output is not None) or (output_method is not None),
                            f"{op} not found as an attribute on either Tensor or the requested module {module}")

            # Accounts for ops that return Tensors, iterables, and other non-Tensors.
            # For example, lstm_cell returns a tuple and equal returns bool.
            def compare(first, second):
                if isinstance(first, torch.Tensor):
                    return torch.equal(first, second)
                elif isinstance(first, collections.abc.Iterable):
                    return all(compare(f, s) for f, s in zip(first, second))
                else:
                    return first == second

            # If both torch.* and Tensor.* variants were found, check outputs are identical
            if (output is not None) and (output_method is not None):
                self.assertTrue(type(output) == type(output_method))
                comparison = compare(output, output_method)
                self.assertTrue(comparison, f"torch.{op} result did not match Tensor.{op} result")

            # Compare numerics to Python-side "autocasting" that (we expect) does the same thing
            # as the C++-side autocasting, and should be bitwise accurate.
            output_to_compare = output if output is not None else output_method
            with torch.cpu.amp.autocast(enabled=False):
                self.assertFalse(torch.is_autocast_cpu_enabled())

                if module is not None and hasattr(module, op):
                    control = getattr(module, op)(*cast(args, run_as_type), **add_kwargs)
                else:
                    control = getattr(args[0].to(run_as_type), op)(*cast(args[1:], run_as_type), **add_kwargs)
                self.assertTrue(type(output_to_compare) == type(control))
                comparison = compare(output_to_compare, control)
                self.assertTrue(comparison, f"torch.{op} result did not match control")
            self.assertTrue(torch.is_autocast_cpu_enabled())
        self.assertFalse(torch.is_autocast_cpu_enabled())

    def args_maybe_kwargs(self, op_with_args):
        if len(op_with_args) == 2:
            return op_with_args[0], op_with_args[1], {}
        else:
            return op_with_args[0], op_with_args[1], op_with_args[2]

    def test_autocast_torch_expect_builtin_promote(self):
        for op, args, out_type in self.autocast_lists.torch_expect_builtin_promote:
            self._run_autocast_outofplace(op, args, torch.float32, out_type=out_type)

    def test_autocast_methods_expect_builtin_promote(self):
        for op, args, out_type in self.autocast_lists.methods_expect_builtin_promote:
            self._run_autocast_outofplace(op, args, torch.float32, module=None, out_type=out_type)

    def test_autocast_torch_bf16(self):
        for op_with_args in self.autocast_lists.torch_bf16:
            op, args, maybe_kwargs = self.args_maybe_kwargs(op_with_args)
            self._run_autocast_outofplace(op, args, torch.bfloat16, add_kwargs=maybe_kwargs)

    def test_autocast_nn_bf16(self):
        for op_with_args in self.autocast_lists.nn_bf16:
            op, args, maybe_kwargs = self.args_maybe_kwargs(op_with_args)
            self._run_autocast_outofplace(op, args, torch.bfloat16, module=torch._C._nn, add_kwargs=maybe_kwargs)

    def test_autocast_torch_fp32(self):
        for op_with_args in self.autocast_lists.torch_fp32:
            op, args, maybe_kwargs = self.args_maybe_kwargs(op_with_args)
            self._run_autocast_outofplace(op, args, torch.float32, add_kwargs=maybe_kwargs)

    def test_autocast_nn_fp32(self):
        for op_with_args in self.autocast_lists.nn_fp32:
            op, args, maybe_kwargs = self.args_maybe_kwargs(op_with_args)
            self._run_autocast_outofplace(op, args, torch.float32, module=torch._C._nn, add_kwargs=maybe_kwargs)

    def test_autocast_torch_need_autocast_promote(self):
        for op, args in self.autocast_lists.torch_need_autocast_promote:
            self._run_autocast_outofplace(op, args, torch.float32)

    @unittest.skipIf(IS_WINDOWS, "Limit support for bf16 path")
    def test_autocast_rnn(self):
        if torch.backends.mkldnn.is_available() and torch.ops.mkldnn._is_mkldnn_bf16_supported():
            x = torch.randn(1, 2, 1)
            hx = torch.randn(2, 2, 1)
            cx = torch.randn(2, 2, 1)

            m = torch.nn.LSTM(1, 1, 2).to(torch.bfloat16)

            # Raise ValueError when autocast is not enabled
            with self.assertRaisesRegex(ValueError, "input must have the type"):
                m(x, (hx, cx))

            # Should be able to run the below case with autocast
            with torch.cpu.amp.autocast():
                m(x, (hx, cx))


class CustomLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_t):
        ctx.save_for_backward(x, w_t)
        return torch.nn.functional.linear(x, w_t)

    @staticmethod
    def backward(ctx, grad_output):
        x, w_t = ctx.saved_tensors
        with torch.autocast(device_type='cuda'):
            dL_dX = torch.matmul(grad_output, w_t)
            dL_dW = torch.matmul(x.transpose(0, 1), grad_output).transpose(0, 1)
        return dL_dX, dL_dW

class WeightDTypeCastCounterMode(TorchDispatchMode):

    def __init__(self, weight):
        super().__init__()
        self.dtype_cast_counter = 0
        self.weight = weight

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if (
            func is torch.ops.aten._to_copy.default and
            args[0] is self.weight and
            kwargs['dtype'] is torch.float16
        ):
            self.dtype_cast_counter += 1
        return func(*args, **kwargs)

    def __enter__(self):
        self.old_clear_cache = torch.clear_autocast_cache
        torch.clear_autocast_cache = lambda: None
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.clear_autocast_cache = self.old_clear_cache
        return super().__exit__(exc_type, exc_val, exc_tb)

@unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
class TestAutocastGPU(TestCase):
    def test_cast_cache_is_global(self):
        """
        Verifies that the autocast cache is global. This is done by
        mocking out cache clearing at the end of the forward pass,
        running forward+backward with an explicit call to autocast in the
        backward, and verifying that the weight only get cast to float16 once.
        """

        data = torch.randn(2, 3).cuda()
        weight = torch.nn.Parameter(torch.randn(4, 3).cuda())

        with WeightDTypeCastCounterMode(weight) as mode:
            with torch.autocast(device_type='cuda'):
                output = CustomLinear.apply(data, weight)
                s = output.sum()
            s.backward()

        self.assertEqual(mode.dtype_cast_counter, 1)

    def test_cache_disabled(self):

        data = torch.randn(2, 3).cuda()
        weight = torch.nn.Parameter(torch.randn(4, 3).cuda())

        try:
            torch._C._set_cached_tensors_enabled(True)
            torch._C._add_cached_tensor(weight)

            with WeightDTypeCastCounterMode(weight) as mode:
                with torch.autocast(device_type='cuda'):
                    output = CustomLinear.apply(data, weight)
                    s = output.sum()
                s.backward()

            # we should not have cached the conversion of the weight
            self.assertEqual(mode.dtype_cast_counter, 2)

        finally:
            torch._C._set_cached_tensors_enabled(False)


class TestTorchAutocast(TestCase):
    def test_autocast_fast_dtype(self):
        gpu_fast_dtype = torch.get_autocast_gpu_dtype()
        cpu_fast_dtype = torch.get_autocast_cpu_dtype()
        self.assertEqual(gpu_fast_dtype, torch.half)
        self.assertEqual(cpu_fast_dtype, torch.bfloat16)

    def test_invalid_device(self):
        dev = 'not a real device'
        msg = f'unsupported autocast device_type \'{dev}\''
        with self.assertRaisesRegex(RuntimeError, msg):
            with torch.autocast(device_type=dev):
                _ = torch.tensor(1)


if __name__ == '__main__':
    
    unittest.main(testRunner=CustomTextTestRunner)

