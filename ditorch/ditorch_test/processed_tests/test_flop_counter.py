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

import torch
from torch.testing._internal.common_utils import TestCase, run_tests, TEST_WITH_TORCHDYNAMO
from torch.testing._internal.common_cuda import SM80OrLater, PLATFORM_SUPPORTS_FUSED_SDPA
import torch.utils.flop_counter
import torch.nn.functional as F
import unittest
import functools


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
try:
    from torchvision import models as torchvision_models
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

HAS_CUDA = torch.cuda.is_available()

def FlopCounterMode(*args, **kwargs):
    return torch.utils.flop_counter.FlopCounterMode(*args, **kwargs, display=False)

def get_total_flops(mode):
    return str(sum([v for _, v in mode.flop_counts["Global"].items()]))

def T(*shape, requires_grad=False):
    return torch.randn(*shape, requires_grad=requires_grad)

@unittest.skipIf(TEST_WITH_TORCHDYNAMO, "torchdynamo doesn't work with __torch_dispatch__ right now")
class TestFlopCounter(TestCase):
    def test_flop_counter_variety(self):
        mode = FlopCounterMode()
        mod = torch.nn.Linear(9, 10)
        with mode:
            torch.mm(T(4, 5), T(5, 6))
            torch.addmm(T(4, 6), T(4, 5), T(5, 6), beta=0.5, alpha=0.5)
            torch.matmul(T(5, 6), T(6, 7))
            torch.einsum("ab,bc->ac", T(6, 7), T(7, 8))
            mod(T(8, 9))

        self.assertExpectedInline(get_total_flops(mode), """3012""")

    def test_op(self):
        mode = FlopCounterMode()
        with mode:
            torch.mm(T(4, 5), T(5, 6))
        # 4 * 6 * 2 * 5 = 240
        self.assertExpectedInline(get_total_flops(mode), """240""")

        with mode:
            torch.bmm(T(3, 4, 5), T(3, 5, 6))
        # 3 * 4 * 6 * 2 * 5 = 720
        self.assertExpectedInline(get_total_flops(mode), """720""")

        with mode:
            torch.addmm(T(4, 6), T(4, 5), T(5, 6))
            torch.addmm(T(4, 1), T(4, 5), T(5, 6))
            torch.addmm(T(6), T(4, 5), T(5, 6))

        # 4 * 6 * 2 * 5 = 240
        self.assertExpectedInline(get_total_flops(mode), """720""")

        with mode:
            torch.baddbmm(T(3, 4, 6), T(3, 4, 5), T(3, 5, 6))

        # 3 * 4 * 6 * 2 * 5 = 720
        self.assertExpectedInline(get_total_flops(mode), """720""")

        with mode:
            torch.conv2d(T(2, 3, 6, 6), T(6, 3, 4, 4), padding=1)

        # out_image_size = 2 * 5 * 5
        # kernel_size = 4 * 4
        # c_out = 6
        # c_in = 3
        # out_image_size * kernel_size * c_out * 2 * c_in

        # NB: I don't think this properly accounts for padding?
        self.assertExpectedInline(get_total_flops(mode), """28800""")

        with mode:
            torch.conv1d(T(2, 3, 6), T(6, 3, 4), padding=1)

        # out_image_size = 2 * 5
        # kernel_size = 4
        # c_out = 6
        # c_in = 3
        # out_image_size * kernel_size * c_out * 2 * c_in

        # NB: I don't think this properly accounts for padding?
        self.assertExpectedInline(get_total_flops(mode), """1440""")

    def test_backward(self):
        mode = FlopCounterMode()
        with mode:
            a = T(4, 5, requires_grad=True)
            a = torch.mm(a, T(5, 6))
            a = a.unsqueeze(0).expand(7, 4, 6)
            a = torch.bmm(a, T(7, 6, 7))
            a.sum().backward()

        self.assertExpectedInline(get_total_flops(mode), """5184""")

    def test_torchscript(self):
        def foo(x):
            return torch.mm(x, x)
        mode = FlopCounterMode()
        with mode:
            foo(T(5, 5))
        unscripted_flops = get_total_flops(mode)
        ts_foo = torch.jit.script(foo)
        with mode:
            ts_foo(T(5, 5))
        self.assertEqual(unscripted_flops, get_total_flops(mode))

    def test_autograd_op(self):
        class _CustomOp(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input: torch.Tensor) -> torch.Tensor:
                return torch.mm(input, input)

            @staticmethod
            def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
                return torch.mm(grad_output, grad_output) + torch.mm(grad_output, grad_output)

        a = T(5, 5, requires_grad=True)
        mode = FlopCounterMode()
        with mode:
            a = _CustomOp.apply(a)
            a.sum().backward()

        self.assertExpectedInline(get_total_flops(mode), """750""")



    @skipIfNoTorchVision
    def test_module(self):
        resnet18 = torchvision_models.resnet18()
        mode = FlopCounterMode(resnet18)
        with mode:
            a = T(1, 3, 224, 224, requires_grad=True)
            resnet18(a).sum().backward()

        self.assertExpectedInline(get_total_flops(mode), """10884440064""")
        layer1_conv_flops = mode.flop_counts['ResNet.layer1'][torch.ops.aten.convolution]
        layer1_conv_back_flops = mode.flop_counts['ResNet.layer1'][torch.ops.aten.convolution_backward]
        self.assertExpectedInline(str(layer1_conv_flops), """924844032""")
        self.assertExpectedInline(str(layer1_conv_back_flops), """1849688064""")

    def test_custom(self):
        mode = FlopCounterMode(custom_mapping={torch.ops.aten.add: lambda *args, out_shape: 5})
        with mode:
            a = T(4, 5)
            a + a

        self.assertExpectedInline(get_total_flops(mode), """5""")

    def test_noop(self):
        mode = FlopCounterMode()
        with mode:
            T(4, 5).cos()

    @unittest.skipIf(not HAS_CUDA, "CUDA not available")
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA or not SM80OrLater, "Does not support SDPA or pre-SM80 hardware")
    def test_sdpa(self):
        batch_size = 4
        n_heads = 8
        seq_len_q = 128
        seq_len_k = 256
        head_dim = 64
        head_dim_v = 64
        dtype = torch.float16

        torch.manual_seed(0)

        def get_flops(batch_size, n_heads, seq_len_q, seq_len_k, head_dim, head_dim_v, dtype, backend, with_backward=False):
            query = torch.randn(batch_size, n_heads, seq_len_q, head_dim, device='cuda', dtype=dtype, requires_grad=True)
            key = torch.randn(batch_size, n_heads, seq_len_k, head_dim, device='cuda', dtype=dtype, requires_grad=True)
            value = torch.randn(batch_size, n_heads, seq_len_k, head_dim_v, device='cuda', dtype=dtype, requires_grad=True)

            if backend == "math":
                backend = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
            elif backend == "flash":
                backend = torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
            elif backend == "mem_efficient":
                backend = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True)

            mode = FlopCounterMode()
            with backend, mode:
                out = F.scaled_dot_product_attention(query, key, value, dropout_p=0, is_causal=True)
                if with_backward:
                    out.sum().backward()
            return int(get_total_flops(mode))

        # Sets seq_len_q == seq_len_k and dim_q == dim_v
        run_uniform_flops = functools.partial(get_flops, batch_size, n_heads, seq_len_q, seq_len_q, head_dim, head_dim, dtype)

        flops = [run_uniform_flops(backend, with_backward=False) for backend in ["math", "flash", "mem_efficient"]]
        flops_fw_math, flops_fw_flash, flops_fw_efficient = flops
        self.assertEqual(flops_fw_math, flops_fw_flash)
        self.assertEqual(flops_fw_math, flops_fw_efficient)

        self.assertExpectedInline(str(flops_fw_math), """134217728""")

        flops = [run_uniform_flops(backend, with_backward=True) for backend in ["math", "flash", "mem_efficient"]]
        flops_fw_bw_math, flops_fw_bw_flash, flops_fw_bw_efficient = flops
        self.assertEqual(flops_fw_math * 3, flops_fw_bw_math)
        self.assertEqual(flops_fw_math * 7 // 2, flops_fw_bw_flash)
        self.assertEqual(flops_fw_bw_flash, flops_fw_bw_efficient)


        run_nonuniform_flops = functools.partial(get_flops, batch_size, n_heads, seq_len_q, seq_len_k, head_dim, head_dim_v, dtype)

        flops = [run_nonuniform_flops(backend, with_backward=False) for backend in ["math", "flash", "mem_efficient"]]
        flops_fw_math, flops_fw_flash, flops_fw_efficient = flops
        self.assertEqual(flops_fw_math, flops_fw_flash, flops_fw_efficient)

        self.assertExpectedInline(str(flops_fw_math), """268435456""")

        flops = [run_nonuniform_flops(backend, with_backward=True) for backend in ["math", "flash", "mem_efficient"]]
        flops_fw_bw_math, flops_fw_bw_flash, flops_fw_bw_efficient = flops
        self.assertExpectedInline(str(flops_fw_bw_math), """805306368""")
        self.assertEqual(flops_fw_bw_flash, flops_fw_bw_efficient)
        self.assertExpectedInline(str(flops_fw_bw_flash), """939524096""")

    def test_hook_registration(self):
        model = torch.nn.Linear(100, 100)
        x = torch.randn(3, 100)

        flop_counter = FlopCounterMode(model)
        with flop_counter:
            self.assertEqual(len(model._forward_pre_hooks), 1)
            self.assertEqual(len(model._forward_hooks), 1)
            model(x).sum().backward()

        self.assertEqual(len(model._forward_pre_hooks), 0)
        self.assertEqual(len(model._forward_hooks), 0)


if __name__ == '__main__':
    
    unittest.main(testRunner=CustomTextTestRunner)

