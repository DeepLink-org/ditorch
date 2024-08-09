import torch
from .base_hook import BaseHook, DisableHookGuard
from .utils import to_device, is_cpu_op
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
        return torch.allclose(a_cpu, b_cpu, atol=atol, rtol=rtol)
    except Exception as e:
        return False
    return False


def compare_result(name, a, b, atol=1e-3):
    error_info = ""
    max_diff = float("nan")
    allclose = False
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        max_diff = tensor_max_diff(a, b)
        allclose = tensor_allclose(a, b)
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
        allclose = a == b
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
                allclose_i = a[i] == b[i]
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
                self.result_device = to_device(
                    "cpu",
                    self.result,
                )
            except Exception as e:
                # some op on cpu backend not support half, bfloat16
                self.args_cpu = to_device(
                    "cpu",
                    self.args_cpu,
                    dtype_cast_dict=OpAutoCompareHook.AUTO_COMPARE_DTYPE_CAST_DICT,
                )
                self.kwargs_cpu = to_device(
                    "cpu",
                    self.kwargs_cpu or {},
                    dtype_cast_dict=OpAutoCompareHook.AUTO_COMPARE_DTYPE_CAST_DICT,
                )
                self.result_cpu = self.func(*self.args_cpu, **self.kwargs_cpu)

                self.result_device = to_device(
                    "cpu",
                    self.result,
                    dtype_cast_dict=OpAutoCompareHook.AUTO_COMPARE_DTYPE_CAST_DICT,
                )
            allclose, max_diff = compare_result(
                self.name, self.result_device, self.result_cpu
            )
            if not allclose and max_diff > 1e-3:
                print(
                    f"OpAutoCompareHook: {self.name:<50} input: {serialize_args_to_dict(*self.args, **self.kwargs)}"
                )
                print(
                    f"OpAutoCompareHook: {self.name:<50} output: {serialize_args_to_dict(self.result)['args']}"
                )
                save_op_args(self.name, "device/input", *self.args, **self.kwargs)
                save_op_args(self.name, "device/output", self.result)
                save_op_args(self.name, "cpu/input", *self.args_cpu, **self.kwargs_cpu)
                save_op_args(self.name, "cpu/output", self.result_cpu)
