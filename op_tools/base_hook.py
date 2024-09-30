# Copyright (c) 2024, DeepLink.
from abc import ABC
from .utils import is_cpu_op, current_location


def is_should_apply_hook(name, func, *args, **kwargs):
    if name is None:
        return False
    if callable(func) is False:
        return False
    if name.startswith("torch.Tensor.") and (name.endswith("__get__") or name.endswith("__set__")):
        return False
    # Assuming that the torch provided by the manufacturer has not been compromised in terms of CPU functionality
    args_on_cpu, device = is_cpu_op(*args, **kwargs)
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
        "torch.Tensor.requires_grad_",
    ]
    if name in EXCLUDE_OPS:
        return False

    return True


class BaseHook(ABC):
    enable = True
    id = 0
    skiped_op = set()
    applied_op = set()

    def __init__(self, name, func) -> None:
        self.name = name
        self.exception = None
        self.func = func
        self.wrapper_func = self.construct_wrapper_func()
        self.condition_funcs = []

    def add_condition_func(self, func):
        if not callable(func):
            raise ValueError("condition_func must be callable")
        self.condition_funcs.append(func)

    def conditions_met(self, *args, **kwargs):
        for func in self.condition_funcs:
            if not func(*args, **kwargs):
                return False
        return True

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
                self.result = self.func(*self.args, **self.kwargs)
            except Exception as e:
                self.result = None
                self.exception = e
            result = self.after_call_op(self.result)
            return result if result is not None else self.result

        return wrapper

    @classmethod
    def class_name(cls):
        return cls.__name__

    def is_should_apply(self, *args, **kwargs):
        return False

    def __call__(self, *args, **kwargs):
        self.args_on_cpu, self.device = is_cpu_op(*args, **kwargs)
        if (
            self.enable
            and self.conditions_met(*args, **kwargs)
            and not self.args_on_cpu
            and is_should_apply_hook(self.name, self.func, *args, **kwargs)
            and self.is_should_apply(*args, **kwargs)
        ):
            self.current_location = current_location(name=self.name, print_stack=False)
            if self.name not in self.applied_op:
                self.applied_op.add(self.name)
                print(f"apply {self.class_name()} on {self.name}")
            self.result = self.wrapper_func(*args, **kwargs)
        else:
            self.result = self.func(*args, **kwargs)
            if not self.args_on_cpu:
                if self.name not in self.skiped_op:
                    self.skiped_op.add(self.name)
                    print(f"skip {self.class_name()} on {self.name}")
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
