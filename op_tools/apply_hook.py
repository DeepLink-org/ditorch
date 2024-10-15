# Copyright (c) 2024, DeepLink.
from torch.overrides import TorchFunctionMode, resolve_name
from .op_capture_hook import OpCaptureHook
from .op_fallback_hook import OpFallbackHook
from .op_autocompare_hook import OpAutoCompareHook
from .op_dispatch_watch_hook import OpDispatchWatcherHook
from .op_time_measure_hook import OpTimeMeasureHook
from .op_dtype_cast_hook import OpDtypeCastHook
from .utils import is_cpu_op
import inspect


def is_should_apply_hook(name, func, args, kwargs=None):
    if name is None:
        return False
    if inspect.isroutine(func) is False:
        return False
    if name.startswith("torch.Tensor.") and (name.endswith("__get__") or name.endswith("__set__")):
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


class OpToolBase(TorchFunctionMode):
    def __init__(self):
        super().__init__()


class OpCapture(OpToolBase):
    """
    Set the OP_CAPTURE_DISABLE_LIST environment variable to ignore specific operators or operators in a specific mode
    Set the OP_CAPTURE_LIST environment variable to only take effect on these operators
    Usage1:
    with OpCapture():
        f()
    Usage2:
    capturer = OpCapture()
    capturer.start()
    f()
    capturer.end()
    """

    def __init__(self):
        super().__init__()

    def __torch_function__(self, func, types, args, kwargs=None):
        name = resolve_name(func)
        hook = OpCaptureHook(name, func)
        return hook(*args, **(kwargs or {}))

    def start(self):
        super().__enter__()

    def stop(self):
        super().__exit__(None, None, None)


class OpFallback(OpToolBase):
    """
    Set the OP_FALLBACK_DISABLE_LIST environment variable to ignore specific operators or operators in a specific mode
    Set the OP_FALLBACK_LIST environment variable to only take effect on these operators
    Usage1:
    with OpFallback():
        f()
    Usage2:
    fallback = OpFallback()
    fallback.start()
    f()
    fallback.end()
    """

    def __torch_function__(self, func, types, args, kwargs=None):
        name = resolve_name(func)
        hook = OpFallbackHook(name, func)
        return hook(*args, **(kwargs or {}))

    def start(self):
        super().__enter__()

    def stop(self):
        super().__exit__(None, None, None)


class OpAutoCompare(OpToolBase):
    """
    Set the OP_AUTOCOMPARE_DISABLE_LIST environment variable to ignore specific operators or operators in a specific mode
    Set the OP_AUTOCOMPARE_LIST environment variable to only take effect on these operators
    Usage1:
    with OpAutocompare():
        f()
    Usage2:
    compare = OpAutocompare()
    compare.start()
    f()
    compare.stop()
    """

    def __torch_function__(self, func, types, args, kwargs=None):
        name = resolve_name(func)
        hook = OpAutoCompareHook(name, func)
        return hook(*args, **(kwargs or {}))

    def start(self):
        super().__enter__()

    def stop(self):
        super().__exit__(None, None, None)


class OpTimeMeasure(OpToolBase):
    """
    Set the OP_TIME_MEASURE_DISABLE_LIST environment variable to ignore specific operators
    Set the OP_TIME_MEASURE_LIST environment variable to only take effect on these operators
    Usage1:
    with OpTimeMeasure():
        f()
    Usage2:
    time_measurer = OpTimeMeasure()
    time_measurer.start()
    f()
    time_measurer.end()
    """

    def __torch_function__(self, func, types, args, kwargs=None):
        name = resolve_name(func)
        hook = OpTimeMeasureHook(name, func)
        return hook(*args, **(kwargs or {}))

    def start(self):
        super().__enter__()

    def stop(self):
        super().__exit__(None, None, None)


class OpDispatchWatcher(OpToolBase):
    """
    Usage1:
    with OpDispatchWatcher():
        f()
    Usage2:
    tool = OpDispatchWatcher()
    tool.start()
    f()
    """

    def __init__(self):
        super().__init__()

    def __torch_function__(self, func, types, args, kwargs=None):
        name = resolve_name(func)
        hook = OpDispatchWatcherHook(name, func)
        return hook(*args, **(kwargs or {}))

    def start(self):
        super().__enter__()

    def stop(self):
        super().__exit__(None, None, None)


class OpDtypeCast(OpToolBase):
    """
    Set the OP_DTYPE_CAST_DISABLE_LIST environment variable to ignore specific operators
    Set the OP_DTYPE_CAST_LIST environment variable to only take effect on these operators
    Usage1:
    with OpDtypeCast():
        f()
    Usage2:
    dtypecaster = OpDtypeCast()
    dtypecaster.start()
    f()
    dtypecaster.end()
    """

    def __init__(self):
        super().__init__()

    def __torch_function__(self, func, types, args, kwargs=None):
        name = resolve_name(func)
        hook = OpDtypeCastHook(name, func)
        return hook(*args, **(kwargs or {}))

    def start(self):
        super().__enter__()

    def stop(self):
        super().__exit__(None, None, None)
