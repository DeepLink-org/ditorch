import torch
from .base_hook import BaseHook, DisableHookGuard
from .utils import to_device, is_cpu_op
from .op_fallback_hook import OpFallbackHook


def compre_obj(a, b):
    if type(a) is not type(b):
        return False, "Inconsistent types"
    if isinstance(a, torch.Tensor):
        atol, rtol = 1e-3, 1e-3
        close = torch.allclose(a, b, atol=atol, rtol=rtol)
        diff = torch.abs(a - b)
        return close, str(diff.max())
    # elif isinstance()


class OpAutoCompareHook(BaseHook):
    def __init__(self, name) -> None:
        super().__init__(name)

    def compare_result(self, device_result, cpu_result):
        self.device_result_ = to_device(device_result)

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
        self.result_device = self.func(*self.args_cpu, **self.kwargs_cpu)
        with DisableHookGuard():
            self.compare_result(self.result_device, self.result)
