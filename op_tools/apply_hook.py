import torch
from torch.overrides import TorchFunctionMode, resolve_name
from torch.utils._python_dispatch import TorchDispatchMode
from .op_capture_hook import OpCaptureHook
from .op_fallback_hook import OpFallbackHook
from .op_dispatch_watch_hook import OpDispatchWatcherHook
import inspect


def is_should_apply_hook(name, func):
    if name is None:
        return False
    if inspect.isroutine(func) == False:
        return False
    if name.startswith("torch.Tensor.__") and name.endswith("__"):
        return False

    return True


class OpCapture(TorchFunctionMode):
    """
    Usage1:
    with OpCapture():
        f()
    Usage2:
    tool = OpCapture()
    tool.__enter__()
    f()
    """

    def is_should_capture(self, name, func):
        if not is_should_apply_hook(name, func):
            return False
        return True

    def __torch_function__(self, func, types, args, kwargs=None):
        name = resolve_name(func)
        if self.is_should_capture(name, func):
            print(f"apply OpCaptureHook on {name}")
            new_func = OpCaptureHook(name)(func)
            return new_func(*args, **(kwargs or {}))
        else:
            print(f"skip OpCaptureHook on {name}")
            return func(*args, **(kwargs or {}))

    def start(self):
        super().__enter__()

    def stop(self):
        super().__exit__(None, None, None)


class OpFallback(TorchFunctionMode):
    """
    Usage1:
    with OpFallback():
        f()
    Usage2:
    tool = OpFallback()
    tool.__enter__()
    f()
    """

    def is_should_fallback(self, name, func):
        if not is_should_apply_hook(name, func):
            return False
        return True

    def __torch_function__(self, func, types, args, kwargs=None):
        name = resolve_name(func)
        if self.is_should_fallback(name, func):
            new_func = OpFallbackHook(name)(func)
            return new_func(*args, **(kwargs or {}))
        else:
            return func(*args, **(kwargs or {}))

    def start(self):
        super().__enter__()

    def stop(self):
        super().__exit__(None, None, None)


class OpDispatchWatcher(TorchDispatchMode):
    """
    Usage1:
    with OpDispatchWatcher():
        f()
    Usage2:
    tool = OpDispatchWatcher()
    tool.__enter__()
    f()
    """

    def __torch_dispatch__(self, func, types, args, kwargs=None):
        name = resolve_name(func)
        if name is not None:
            new_func = OpDispatchWatcherHook(name)(func)
            return new_func(*args, **(kwargs or {}))
        else:
            return func(*args, **(kwargs or {}))

    def start(self):
        super().__enter__()

    def stop(self):
        super().__exit__(None, None, None)
