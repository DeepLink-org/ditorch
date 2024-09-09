import inspect
import torch
from torch.overrides import resolve_name
from .utils import traverse_container, get_function_from_string
from .op_capture_hook import OpCaptureHook
from .op_fallback_hook import OpFallbackHook
from .op_autocompare_hook import OpAutoCompareHook
from .op_dispatch_watch_hook import OpDispatchWatcherHook
from .op_time_measure_hook import OpTimeMeasureHook
from .op_dtype_cast_hook import OpDtypeCastHook


def get_func_name(func):
    try:
        name = resolve_name(func)
    except Exception:
        name = None
    if name is None:
        if hasattr(func, "__module__"):
            name = func.__module__ + "."
            if hasattr(func, "__name__"):
                name += func.__name__
    if name is None:
        name = str(func)
    return name


def apply_hook_to_ops(ops, hook):
    for op in traverse_container(ops):
        if isinstance(op, str):
            func = get_function_from_string(op)
            name = op
            module = get_function_from_string(op[: op.rfind(".")])
        elif inspect.isroutine(op):
            func = op
            name = get_func_name(func)
            module = inspect.getmodule(func)
        elif inspect.ismodule(op):
            for name, obj in inspect.getmembers(op, inspect.isroutine):
                apply_hook_to_ops(obj, hook)
            continue
        else:
            continue
        if func is None or module is None:
            print(f"can not apply {hook.__name__} to {op}")
            continue
        if not hasattr(func, "__name__"):
            func.__name__ = name.split(".")[-1]
        hook_obj = hook(name, func)
        setattr(module, func.__name__, hook_obj)


def fallback_ops(ops):
    apply_hook_to_ops(ops, OpFallbackHook)


def fallback_op_if(op, condition=lambda *args, **kwargs: False):
    apply_hook_to_ops(op, OpFallbackHook)


def dump_ops_args(ops):
    apply_hook_to_ops(ops, OpDispatchWatcherHook)


def dump_all_ops_args():
    apply_hook_to_ops(torch, OpDispatchWatcherHook)


def autocompare_ops(ops):
    apply_hook_to_ops(ops, OpAutoCompareHook)


def measure_ops_elasped(ops):
    apply_hook_to_ops(ops, OpTimeMeasureHook)
