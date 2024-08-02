import torch
from torch.overrides import TorchFunctionMode, resolve_name
from torch.utils._python_dispatch import TorchDispatchMode
from .op_capture_hook import OpCaptureHook
from .op_fallback_hook import OpFallbackHook
from .op_autocompare_hook import OpAutoCompareHook
from .op_dispatch_watch_hook import OpDispatchWatcherHook
from .utils import is_cpu_op
import inspect


def is_should_apply_hook(name, func, args, kwargs=None):
    if name is None:
        return False
    if inspect.isroutine(func) == False:
        return False
    # if name.startswith("torch.Tensor.__") and name.endswith("__"):
    #    return False
    # Assuming that the torch provided by the manufacturer has not been compromised in terms of CPU functionality
    args_on_cpu, device = is_cpu_op(args, kwargs)
    if args_on_cpu:
        return False

    EXCLUDE_OPS = [
        "torch.Tensor.backward",
        "torch.Tensor.numel",
        "torch.Tensor.size",
        "torch.Tensor.__repr__",
    ]
    if name in EXCLUDE_OPS:
        return False

    return True


class OpCapture(TorchFunctionMode):
    """
    Usage1:
    with OpCapture():
        f()
    Usage2:
    capturer = OpCapture()
    capturer.start()
    f()
    capturer.end()
    """

    def is_should_capture(self, name, func, args, kwargs=None):
        if not is_should_apply_hook(name, func, args, kwargs=None):
            return False

        return True

    def __torch_function__(self, func, types, args, kwargs=None):
        name = resolve_name(func)
        if self.is_should_capture(name, func, args, kwargs):
            print(f"apply OpCaptureHook on {name} {types if len(types) > 0 else ''}")
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
    fallback = OpFallback()
    fallback.start()
    f()
    fallback.end()
    """

    def is_should_fallback(self, name, func, args, kwargs=None):
        if not is_should_apply_hook(name, func, args, kwargs):
            return False
        return True

    def __torch_function__(self, func, types, args, kwargs=None):
        name = resolve_name(func)
        if self.is_should_fallback(name, func, args, kwargs):
            new_func = OpFallbackHook(name)(func)
            return new_func(*args, **(kwargs or {}))
        else:
            return func(*args, **(kwargs or {}))

    def start(self):
        super().__enter__()

    def stop(self):
        super().__exit__(None, None, None)


class OpAutocompare(TorchFunctionMode):
    """
    Usage1:
    with OpAutocompare():
        f()
    Usage2:
    compare = OpAutocompare()
    compare.start()
    f()
    compare.stop()
    """

    def is_should_compare(self, name, func, args, kwargs=None):
        if not is_should_apply_hook(name, func, args, kwargs):
            return False
        return True

    def __torch_function__(self, func, types, args, kwargs=None):
        name = resolve_name(func)
        if self.is_should_compare(name, func, args, kwargs):
            new_func = OpAutoCompareHook(name)(func)
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
    tool.start()
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
