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


def apply_hook_to_ops(ops, hook, condition_funcs=[]):
    index = -1
    for op in traverse_container(ops):
        index += 1
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
        if isinstance(condition_funcs, list):
            if len(condition_funcs) > index:
                condition_func = condition_funcs[index]
            else:
                condition_func = lambda *args, **kwargs: True
        else:
            condition_func = condition_funcs
        assert callable(condition_func)

        hook_obj = hook(name, func)
        hook_obj.add_condition_func(condition_func)
        setattr(module, func.__name__, hook_obj)


def apply_feature(ops, feature, condition_func=lambda *args, **kwargs: True):
    assert isinstance(ops, (str, list))
    feature_options = ["fallback", "autocompare", "op_time_measure", "dump_op_args"]
    assert (
        feature in feature_options
    ), f"feature must be one of {feature_options}, but got {feature}"
    assert callable(condition_func)
    if feature == "fallback":
        hook_cls = OpFallbackHook
    elif feature == "autocompare":
        hook_cls = OpAutoCompareHook
    elif feature == "op_time_measure":
        hook_cls = OpTimeMeasureHook
    elif feature == "dump_op_args":
        hook_cls = OpDispatchWatcherHook

    if isinstance(ops, str):
        apply_hook_to_ops(ops, hook_cls, condition_func)
    elif isinstance(ops, list):
        for op in ops:
            apply_hook_to_ops(op, hook_cls, condition_func)
    else:
        assert False, f"ops must be str or list, but got {type(ops)}"
