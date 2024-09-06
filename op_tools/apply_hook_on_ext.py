import inspect
import torch
from .utils import traverse_container, get_function_from_string
from .op_capture_hook import OpCaptureHook
from .op_fallback_hook import OpFallbackHook
from .op_autocompare_hook import OpAutoCompareHook
from .op_dispatch_watch_hook import OpDispatchWatcherHook
from .op_time_measure_hook import OpTimeMeasureHook
from .op_dtype_cast_hook import OpDtypeCastHook


def apply_hook_to_ops(ops, hook):
    for op in traverse_container(ops):
        if isinstance(op, str):
            func = get_function_from_string(op)
            name = op
        elif inspect.isroutine(op):
            func = op
            if hasattr(func, "__module__"):
                name = func.__module__ + "." + func.__name__
        elif inspect.ismodule(op):
            for name, obj in inspect.getmembers(op, inspect.isroutine):
                apply_hook_to_ops(obj, hook)
                continue
            return
        else:
            func = None
        module = inspect.getmodule(func) if func is not None else None
        if module is None:
            print(f"can not apply {hook.__name__} to {op}")
            continue
        hook_obj = hook(name, func)
        setattr(module, func.__name__, hook_obj)
        print(f"{hook.__name__} applied to ", name)


def fallback_ops(ops):
    apply_hook_to_ops(ops, OpFallbackHook)


def dump_ops_args(ops):
    apply_hook_to_ops(ops, OpDispatchWatcherHook)


def dump_all_ops_args():
    apply_hook_to_ops(torch, OpDispatchWatcherHook)


def autocompare_ops(ops):
    apply_hook_to_ops(ops, OpAutoCompareHook)


def measure_ops_elasped(ops):
    apply_hook_to_ops(ops, OpTimeMeasureHook)
