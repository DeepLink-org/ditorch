import torch
from .base_hook import BaseHook, DisableHookGuard


def is_cpu_op(*args, **kwargs):
    device = "cpu"
    for v in args:
        if isinstance(v, torch.Tensor):
            if v.is_cpu:
                return True, "cpu"
            else:
                device = v.device
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            if v.is_cpu:
                return True, "cpu"
            else:
                device = v.device
    return False, device


def transform_args_to_device1(device, *args, **kwargs):
    if args is None and kwargs is None:
        return None

    def to_device(obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        if isinstance(obj, (tuple, list)):
            return type(obj)([to_device(v) for v in obj])
        if isinstance(obj, dict):
            return {k: to_device(v) for k, v in obj}
        else:
            return obj

    args_transformed = tuple(to_device(arg) for arg in args)
    kwargs_transformed = {k: to_device(v) for k, v in kwargs.items()}

    return args_transformed, kwargs_transformed


def to_device(device, obj):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, (tuple, list)):
        return type(obj)([to_device(device, v) for v in obj])
    elif isinstance(obj, dict):
        return {k: to_device(device, v) for k, v in obj}
    else:
        return obj


class OpFallbackHook(BaseHook):
    def __init__(self, name) -> None:
        super().__init__(name)

    def before_call_op(self, *args, **kwargs):

        with DisableHookGuard():
            self.is_cpu_op, self.device = is_cpu_op(*args, **kwargs)
            if self.is_cpu_op:
                return

            self.args_device = self.args
            self.kwargs_device = self.kwargs or {}
            self.args = to_device("cpu", self.args)
            self.kwargs = to_device("cpu", self.kwargs or {})

    def after_call_op(self, result):
        if self.is_cpu_op:
            return
        with DisableHookGuard():
            self.result_device = to_device(self.device, self.result)
