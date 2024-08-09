import os
import torch
from torch.overrides import TorchFunctionMode, resolve_name
from torch.utils._python_dispatch import TorchDispatchMode
from .op_capture_hook import OpCaptureHook
from .op_fallback_hook import OpFallbackHook
from .op_autocompare_hook import OpAutoCompareHook
from .op_dispatch_watch_hook import OpDispatchWatcherHook
from .op_time_measure_hook import OpTimeMeasureHook
from .utils import is_cpu_op, is_opname_match
import inspect


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


class OpCapture(TorchFunctionMode):
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

    def is_should_capture(self, name, func, args, kwargs=None):
        if not is_should_apply_hook(name, func, args, kwargs=None):
            return False

        if is_opname_match(name, os.getenv("OP_CAPTURE_DISABLE_LIST", "")):
            return False

        return is_opname_match(name, os.getenv("OP_CAPTURE_LIST", ".*"))

    def __torch_function__(self, func, types, args, kwargs=None):
        name = resolve_name(func)
        if self.is_should_capture(name, func, args, kwargs):
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


VIEW_OPS = [
    "torch.Tensor.reshape",
    "torch.Tensor.adjoint",
    "torch.Tensor.as_strided",
    "torch.Tensor.detach",
    "torch.Tensor.diagonal",
    "torch.Tensor.expand",
    "torch.Tensor.expand_as",
    "torch.Tensor.movedim",
    "torch.Tensor.narrow",
    "torch.Tensor.permute",
    "torch.Tensor.select",
    "torch.Tensor.squeeze",
    "torch.Tensor.transpose",
    "torch.Tensor.t",
    "torch.Tensor.T",
    "torch.Tensor.H",
    "torch.Tensor.mT",
    "torch.Tensor.mH",
    "torch.Tensor.real",
    "torch.Tensor.imag",
    "torch.Tensor.view_as_real",
    "torch.Tensor.unflatten",
    "torch.Tensor.unfold",
    "torch.Tensor.unsqueeze",
    "torch.Tensor.view",
    "torch.Tensor.view_as",
    "torch.Tensor.unbind",
    "torch.Tensor.split",
    "torch.Tensor.hsplit",
    "torch.Tensor.vsplit",
    "torch.Tensor.tensor_split",
    "torch.Tensor.split_with_sizes",
    "torch.Tensor.swapaxes",
    "torch.Tensor.swapdims",
    "torch.Tensor.chunk",
    "torch.Tensor.indices",
    "torch.Tensor.values",
]


class OpFallback(TorchFunctionMode):
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

    def is_should_fallback(self, name, func, args, kwargs=None):
        if not is_should_apply_hook(name, func, args, kwargs):
            return False
        if is_opname_match(name, os.getenv("OP_FALLBACK_DISABLE_LIST", "")):
            return False
        # if name in VIEW_OPS:
        #    return False

        return is_opname_match(name, os.getenv("OP_FALLBACK_LIST", ".*"))

    def __torch_function__(self, func, types, args, kwargs=None):
        name = resolve_name(func)
        if self.is_should_fallback(name, func, args, kwargs):
            new_func = OpFallbackHook(name)(func)
            return new_func(*args, **(kwargs or {}))
        else:
            print(f"skip OpFallbackHook on {name}")
            return func(*args, **(kwargs or {}))

    def start(self):
        super().__enter__()

    def stop(self):
        super().__exit__(None, None, None)


RANDOM_NUMBER_GEN_OPS = [
    "torch.Tensor.random_",
    "torch.randperm",
    "torch.bernoulli",
    "torch.poisson",
    "torch.randint_like",
    "torch.randint",
    "torch.randn",
    "torch.randn_like",
    "torch.multinomial",
    "torch.nn.init.kaiming_uniform",
    "torch.nn.init.kaiming_uniform_",
    "torch.nn.init.trunc_normal_",
    "torch.nn.init.uniform",
    "torch.nn.init.normal",
    "torch.nn.init.uniform_",
    "torch.nn.init.normal_",
    "torch.nn.init.warnings",
    "torch.nn.init.xavier_normal",
    "torch.nn.init.xavier_normal_",
    "torch.nn.init.xavier_uniform",
    "torch.nn.init.kaiming_normal",
    "torch.nn.init.xavier_uniform_",
    "torch.nn.init.kaiming_normal_",
]


class OpAutoCompare(TorchFunctionMode):
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

    def is_should_compare(self, name, func, args, kwargs=None):
        if not is_should_apply_hook(name, func, args, kwargs):
            return False
        if is_opname_match(name, os.getenv("OP_AUTOCOMPARE_DISABLE_LIST", "")):
            return False

        if name in RANDOM_NUMBER_GEN_OPS:
            return False

        return is_opname_match(name, os.getenv("OP_AUTOCOMPARE_LIST", ".*"))

    def __torch_function__(self, func, types, args, kwargs=None):
        name = resolve_name(func)
        if self.is_should_compare(name, func, args, kwargs):
            new_func = OpAutoCompareHook(name)(func)
            return new_func(*args, **(kwargs or {}))
        else:
            print(f"skip OpAutoCompareHook on {name}")
            return func(*args, **(kwargs or {}))

    def start(self):
        super().__enter__()

    def stop(self):
        super().__exit__(None, None, None)


class OpTimeMeasure(TorchFunctionMode):
    """
    Set the OP_TIME_MEASURE_DISABLE_LIST environment variable to ignore specific operators or operators in a specific mode
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

    def is_should_measure(self, name, func, args, kwargs=None):
        if not is_should_apply_hook(name, func, args, kwargs=None):
            return False

        if is_opname_match(name, os.getenv("OP_TIME_MEASURE_DISABLE_LIST", "")):
            return False

        return is_opname_match(name, os.getenv("OP_TIME_MEASURE_LIST", ".*"))

    def __torch_function__(self, func, types, args, kwargs=None):
        name = resolve_name(func)
        if self.is_should_measure(name, func, args, kwargs):
            print(f"apply OpTimeMeasureHook on {name}")
            new_func = OpTimeMeasureHook(name)(func)
            return new_func(*args, **(kwargs or {}))
        else:
            print(f"skip OpTimeMeasureHook on {name}")
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
