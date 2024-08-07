import torch
from .base_hook import BaseHook, DisableHookGuard
from .utils import to_device, is_cpu_op
from .op_fallback_hook import OpFallbackHook
from .save_op_args import save_op_args


def compre_obj(a, b):
    # We assume they are of the same type: torch.nn.parameter.Parameter and torch.Tensor
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        atol, rtol = 1e-3, 1e-3
        a_cpu, b_cpu = a.cpu(), b.cpu()
        if a_cpu.dtype == torch.bool:
            a_cpu = a_cpu.int()
        if b_cpu.dtype == torch.bool:
            b_cpu = b_cpu.int()
        diff = torch.abs(a_cpu - b_cpu)
        max_diff = diff.max().item()
        if a_cpu.dtype == b_cpu.dtype:
            return max_diff
        else:
            return f"Inconsistent dtypes: {a.dtype} {b.dtype}, max_diff:{max_diff}"
    elif type(a) is not type(b):
        return f"Inconsistent types: {type(a)} {type(b)}"
    elif isinstance(a, (list, tuple)):
        return type(a)([compre_obj(a[i], b[i]) for i in range(len(a))])
    elif isinstance(a, dict):
        return {k: compre_obj(a[k], b[k]) for k in a.keys()}
    elif isinstance(a, (int, float, complex)):
        return abs(a - b)
    elif a is None:
        return 0.0
    else:
        return f"{__file__} unhandle type:{a} {b}"


class OpAutoCompareHook(BaseHook):
    def __init__(self, name) -> None:
        super().__init__(name)

    def compare_result(self, device_result, cpu_result):
        self.compare_result = compre_obj(device_result, cpu_result)
        allclose = True
        if isinstance(self.compare_result, (int, float, complex)):  # f"{max_diff:.9f}"
            print(
                f"OpAutoCompareHook: {self.name:<50} max_diff: {f'{self.compare_result:20.9f}'}"
            )
            if self.compare_result > 1e-3:
                allclose = False
        elif isinstance(self.compare_result, (list, tuple)):
            for i in range(len(self.compare_result)):
                print(
                    f"OpAutoCompareHook: {self.name:<50} {i}th \tmax_diff: {f'{self.compare_result[i]:20.9f}'}"
                )
                if self.compare_result[i] > 1e-3:
                    allclose = False
        elif isinstance(self.compare_result, (dict,)):
            for k, v in self.compare_result.items():
                print(
                    f"OpAutoCompareHook: {self.name:<50} {k} \tmax_diff: {f'{v:20.9f}'}"
                )
                if v > 1e-3:
                    allclose = False
        else:
            print(
                f"OpAutoCompareHook: {self.name:<50} compare_result: {self.compare_result}"
            )

        if not allclose:
            save_op_args(self.name, "device/input", *self.args, **self.kwargs)
            save_op_args(self.name, "device/output", self.result)
            save_op_args(self.name, "cpu/input", *self.args_cpu, **self.kwargs_cpu)
            save_op_args(self.name, "cpu/output", self.result_cpu)

    def before_call_op(self, *args, **kwargs):
        with DisableHookGuard():
            self.is_cpu_op, self.device = is_cpu_op(*args, **kwargs)
            if self.is_cpu_op:
                return
            self.args_cpu = to_device("cpu", self.args)
            self.kwargs_cpu = to_device("cpu", self.kwargs or {})

    def after_call_op(self, result):
        if self.is_cpu_op:
            return
        with DisableHookGuard():
            self.result_cpu = self.func(*self.args_cpu, **self.kwargs_cpu)
            self.compare_result(self.result, self.result_cpu)
