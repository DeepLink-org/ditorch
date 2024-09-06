# Copyright (c) 2024, DeepLink.
from abc import ABC, abstractmethod
import inspect
from .utils import is_cpu_op, is_opname_match


def is_should_apply_hook(name, func, args, kwargs=None):
    if name is None:
        return False
    if inspect.isroutine(func) == False:
        return False
    if name.startswith("torch.Tensor.") and (
        name.endswith("__get__") or name.endswith("__set__")
    ):
        return False
    # Assuming that the torch provided by the manufacturer has not been compromised in terms of CPU functionality
    args_on_cpu, device = is_cpu_op(args, kwargs)
    if args_on_cpu:
        return False

    EXCLUDE_OPS = [
        "torch.Tensor.data_ptr",
        "torch.Tensor.backward",
        "torch.Tensor.has_names",
        "torch.Tensor.numel",
        "torch.Tensor.size",
        "torch.Tensor.__repr__",
        "torch.Tensor.__format__",
        "torch.Tensor.type",
        "torch.Tensor.dim",
    ]
    if name in EXCLUDE_OPS:
        return False

    return True


class BaseHook(ABC):
    enable = True
    id = 0

    def __init__(self, name, func) -> None:
        self.name = name
        self.exception = None
        self.func = func
        self.wrapper_func = self.construct_wrapper_func()

    def before_call_op(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def after_call_op(self, result):
        self.result = result

    def construct_wrapper_func(self):
        def wrapper(*args, **kwargs):
            BaseHook.id = BaseHook.id + 1
            self.args = args
            self.kwargs = kwargs
            self.before_call_op(*args, **kwargs)
            try:
                result = self.func(*self.args, **self.kwargs)
            except Exception as e:
                result = None
                self.exception = e
            return result

        return wrapper

    def is_should_aply(self, *args, **kwargs):
        return True

    def __call__(self, *args, **kwargs):
        if (
            self.enable
            and is_should_apply_hook(self.name, self.func, args, kwargs)
            and self.is_should_apply(*args, **kwargs)
        ):
            self.result = self.wrapper_func(*self.args, **self.kwargs)
        else:
            self.result = self.func(*args, **kwargs)
        return self.result


class DisableHookGuard:
    counter = 0

    def __init__(self) -> None:
        pass

    def __enter__(self):
        DisableHookGuard.counter = DisableHookGuard.counter + 1
        BaseHook.enable = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        DisableHookGuard.counter = DisableHookGuard.counter - 1
        if DisableHookGuard.counter <= 0:
            BaseHook.enable = True
