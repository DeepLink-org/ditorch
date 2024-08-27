# Copyright (c) 2024, DeepLink.
from abc import ABC, abstractmethod


class BaseHook(ABC):
    enable = True
    id = 0

    def __init__(self, name) -> None:
        self.name = name
        self.exception = None

    def before_call_op(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def after_call_op(self, result):
        self.args = None
        self.kwargs = None
        self.result = result

    def __call__(self, func):
        self.func = func

        def wrapper(*args, **kwargs):
            if self.enable:
                BaseHook.id = BaseHook.id + 1
                self.args = args
                self.kwargs = kwargs
                self.before_call_op(*args, **kwargs)
                try:
                    self.result = self.func(*self.args, **self.kwargs)
                except Exception as e:
                    self.result = None
                    self.exception = e
                self.after_call_op(self.result)
            else:
                self.result = func(*args, **kwargs)
            return self.result

        return wrapper


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
