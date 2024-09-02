# Copyright (c) 2024, DeepLink.
import torch
import math
import gc

from .base_hook import BaseHook, DisableHookGuard
from .utils import to_device, is_cpu_op, traverse_container, is_inplace_op
from .op_fallback_hook import OpFallbackHook
from .save_op_args import save_op_args, serialize_args_to_dict


def tensor_max_diff(a, b):
    a_cpu, b_cpu = a.cpu(), b.cpu()
    if a_cpu.dtype == torch.bool:
        a_cpu = a_cpu.int()
    if b_cpu.dtype == torch.bool:
        b_cpu = b_cpu.int()
    diff = torch.abs(a_cpu - b_cpu)
    max_diff = diff.max().item()
    return max_diff


def tensor_allclose(a, b, atol=1e-3, rtol=1e-3):
    a_cpu, b_cpu = a.cpu(), b.cpu()
    try:
        return torch.allclose(a_cpu, b_cpu, atol=atol, rtol=rtol, equal_nan=True)
    except Exception as e:
        return False
    return False


def compare_result(name, a, b, atol=1e-3):
    error_info = ""
    max_diff = float("nan")
    allclose = False
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape == b.shape:
            max_diff = tensor_max_diff(a, b)
            allclose = tensor_allclose(a, b)
        else:
            max_diff = float("nan")
            allclose = False
            error_info = f"Inconsistent shape: {a.shape} {b.shape}"
        if a.dtype != b.dtype:
            error_info = f"Inconsistent dtypes: {a.dtype} {b.dtype}"
        print(
            f"OpAutoCompareHook: {name:<50} allclose: {allclose}\tmax_diff: {f'{max_diff:20.9f}'} {error_info}"
        )
    elif type(a) != type(b):
        error_info = f"Inconsistent types: {a} {b}"
        print(
            f"OpAutoCompareHook: {name:<50} allclose: {allclose}\tmax_diff: {f'{max_diff:20.9f}'} {error_info}"
        )
    elif isinstance(a, (bool, int, float)):
        allclose = a == b or (math.isnan(a) and math.isnan(b))
        max_diff = a - b
        print(
            f"OpAutoCompareHook: {name:<50} allclose: {allclose}\tmax_diff: {f'{max_diff:20.9f}'}"
        )
    elif type(a).__module__.startswith("torch.return_types") or isinstance(
        a, (tuple, list)
    ):
        max_diff_list = []
        allclose_list = []
        error_info_i = ""
        for i in range(len(a)):
            if isinstance(a[i], torch.Tensor) and isinstance(a[i], torch.Tensor):
                max_diff_i = tensor_max_diff(a[i], b[i])
                allclose_i = tensor_allclose(a[i], b[i])
                max_diff_list.append(max_diff_i)
                allclose_list.append(allclose_i)
                if a[0].dtype != b[0].dtype:
                    error_info_i = f"Inconsistent dtypes: {a[i].dtype} {b[i].dtype}"
                print(
                    f"OpAutoCompareHook: {name:<46} {i}th allclose: {allclose_i}\tmax_diff: {f'{max_diff_i:20.9f}'} {error_info_i}"
                )
            else:
                allclose_i = a[i] == b[i] or (math.isnan(a[i]) and math.isnan(b[i]))
                max_diff_i = a[i] - b[i]
                max_diff_list.append(max_diff_i)
                allclose_list.append(allclose_i)
                print(
                    f"OpAutoCompareHook: {name:<46} {i}th allclose: {allclose_i}\tmax_diff: {f'{max_diff_i:20.9f}'} {error_info_i}"
                )

        allclose = all(allclose_list)
        max_diff = max(max_diff_list)
    else:
        print(f"OpAutoCompareHook: {name:} {__file__} unhandle output type: {type(a)}")

    return allclose, max_diff


class BackwardHookHandle:
    def __init__(self, compare_hook) -> None:
        self.compare_hook = compare_hook

    def register_grad_fn_hook(self, tensor):
        hook_handle = None

        def grad_fun(grad_inputs, grad_outputs):
            self.compare_hook.run_backward_on_cpu(grad_inputs, grad_outputs)
            self.compare_hook.compare_all_grad()
            hook_handle.remove()

        hook_handle = tensor.grad_fn.register_hook(grad_fun)
        return grad_fun

    def register_tensor_hook(self, index, tensor):
        hook_handle = None

        def grad_fun(grad):
            self.compare_hook.set_input_grad(index, grad)
            self.compare_hook.compare_all_grad()
            hook_handle.remove()

        hook_handle = tensor.register_hook(grad_fun)

        return grad_fun


class OpAutoCompareHook(BaseHook):
    AUTO_COMPARE_DTYPE_CAST_DICT = {
        torch.half: torch.float32,
        torch.bfloat16: torch.float32,
    }

    def __init__(self, name) -> None:
        super().__init__(name)

    def before_call_op(self, *args, **kwargs):
        with DisableHookGuard():
            self.is_cpu_op, self.device = is_cpu_op(*args, **kwargs)
            if self.is_cpu_op:
                return
            self.args_cpu = to_device(
                "cpu",
                self.args,
            )
            self.kwargs_cpu = to_device(
                "cpu",
                self.kwargs or {},
            )

    def after_call_op(self, result):
        if self.is_cpu_op:
            return
        with DisableHookGuard():
            try:
                self.result_cpu = self.func(*self.args_cpu, **self.kwargs_cpu)
                self.result_device = to_device("cpu", self.result)
                self.dtype_cast_dict = dict()
            except Exception as e:
                self.dtype_cast_dict = OpAutoCompareHook.AUTO_COMPARE_DTYPE_CAST_DICT
                # some op on cpu backend not support half, bfloat16
                self.args_cpu = to_device(
                    "cpu",
                    self.args_cpu,
                    dtype_cast_dict=self.dtype_cast_dict,
                )
                self.kwargs_cpu = to_device(
                    "cpu",
                    self.kwargs_cpu or {},
                    dtype_cast_dict=self.dtype_cast_dict,
                )
                # RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
                if is_inplace_op(self.name) and self.args[0].requires_grad:
                    print(f"inplace op")
                    args_cpu = [item for item in self.args_cpu]
                    args_cpu[0] = args_cpu[0].clone()
                    self.result_cpu = self.func(*args_cpu, **self.kwargs_cpu)
                else:
                    self.result_cpu = self.func(*self.args_cpu, **self.kwargs_cpu)

                self.result_device = to_device(
                    "cpu",
                    self.result,
                    dtype_cast_dict=self.dtype_cast_dict,
                )
            allclose, max_diff = compare_result(
                self.name, self.result_device, self.result_cpu
            )
            if not allclose and max_diff > 1e-3:
                print(
                    f"OpAutoCompareHook: {self.name:<60} input: {serialize_args_to_dict(*self.args, **self.kwargs)}"
                )
                print(
                    f"OpAutoCompareHook: {self.name:<60} output: {serialize_args_to_dict(self.result)['args']}"
                )
                save_op_args(self.name, "device/input", *self.args, **self.kwargs)
                save_op_args(self.name, "device/output", self.result)
                save_op_args(self.name, "cpu/input", *self.args_cpu, **self.kwargs_cpu)
                save_op_args(self.name, "cpu/output", self.result_cpu)

            self.backward_hook_handle = BackwardHookHandle(self)
            for result in traverse_container(self.result):
                if isinstance(result, torch.Tensor):
                    if result.grad_fn is not None:
                        self.backward_hook_handle.register_grad_fn_hook(result)
            index = -1
            for arg in traverse_container(self.args):
                index += 1
                if isinstance(arg, torch.Tensor) and arg.requires_grad:
                    self.backward_hook_handle.register_tensor_hook(index, arg)

    def run_backward_on_cpu(self, grad_inputs, grad_output):
        self.grad_inputs = grad_inputs
        self.grad_output = grad_output
        self.grad_outputs_cpu = to_device("cpu", grad_output, self.dtype_cast_dict)
        self.grad_inputs_cpu = to_device("cpu", grad_inputs, self.dtype_cast_dict)
        for arg_cpu in traverse_container(self.args_cpu):
            if isinstance(arg_cpu, torch.Tensor) and arg_cpu.grad is not None:
                arg_cpu.grad.zero_()

        for result_cpu in traverse_container(self.result_cpu):
            if isinstance(result_cpu, torch.Tensor) and result_cpu.requires_grad:
                result_cpu.backward(*self.grad_outputs_cpu)

        self.args_cpu_grad = []
        for i in range(len(self.args_cpu)):
            if (
                isinstance(self.args_cpu[i], torch.Tensor)
                and self.args_cpu[i].grad is not None
            ):
                self.args_cpu_grad.append(self.args_cpu[i].grad)
            else:
                self.args_cpu_grad.append(None)

    def count_params_with_requires_grad(self):
        count = 0
        for arg in traverse_container(self.args):
            if isinstance(arg, torch.Tensor) and arg.requires_grad:
                count = count + 1
        return count

    def compare_all_grad(self):
        """
        Since the order in which the two backward hooks registered by different operators are called is not exactly the same, and the gradient can only be calculated on the CPU after the grad_fn hook is called, there is a judgment here, because only when the gradient on the CPU is calculated and all the gradients of the device parameters are obtained, can the gradient information be compared.
        """
        if not hasattr(self, "args_cpu_grad"):
            return
        if not hasattr(self, "args_grad"):
            return

        # Check if all gradients have been obtained
        for i in range(len(self.args)):
            arg = self.args[i]
            if isinstance(arg, torch.Tensor) and (
                arg.requires_grad and self.args_grad[i] is None
            ):
                return

        for i in range(len(self.args)):
            arg = self.args[i]

            if isinstance(arg, torch.Tensor) and arg.requires_grad:
                arg_cpu_grad = self.args_cpu_grad[i]
                allclose, max_diff = compare_result(
                    self.name + f" (ins[{i}].grad)",
                    self.args_grad[i],
                    arg_cpu_grad,
                )
                if not allclose and max_diff > 1e-3:
                    print(f"{self.name} {i}th grad is not allclose ")

        gc.collect()

    def set_input_grad(self, index, grad):
        if not hasattr(self, "args_grad"):
            self.args_grad = [None for i in range(len(self.args))]
        self.args_grad[index] = to_device("cpu", grad, self.dtype_cast_dict)
