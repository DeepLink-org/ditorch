import torch
from .base_hook import BaseHook, DisableHookGuard
from .utils import to_device, is_cpu_op


class OpFallbackHook(BaseHook):
    def __init__(self, name) -> None:
        super().__init__(name)

    def before_call_op(self, *args, **kwargs):
        with DisableHookGuard():
            self.is_cpu_op, self.device = is_cpu_op(*args, **kwargs)
            if self.is_cpu_op:
                return

            self.args = to_device("cpu", self.args)
            self.kwargs = to_device("cpu", self.kwargs or {})

    def after_call_op(self, result):
        if self.is_cpu_op:
            return
        with DisableHookGuard():
            self.result = to_device(self.device, self.result)
