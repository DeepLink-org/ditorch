import inspect
from torch.overrides import resolve_name
from .utils import traverse_container, get_function_from_string
from .op_capture_hook import OpCaptureHook
from .op_fallback_hook import OpFallbackHook
from .op_autocompare_hook import OpAutoCompareHook
from .op_observe_hook import OpObserveHook
from .op_time_measure_hook import OpTimeMeasureHook
from .op_dtype_cast_hook import OpDtypeCastHook
from .op_overflow_check_hook import OpOverflowCheckHook
from .base_hook import BaseHook


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

                def condition_func(*args, **kwargs):
                    return True

        else:
            condition_func = condition_funcs
        assert callable(condition_func)
        if issubclass(type(func), BaseHook):
            print(
                f"The {name} is applying multiple hooks, and the previous hook {func.class_name()} will be replaced by the {hook.class_name()}."  # noqa: E501
            )
            func = func.func

        hook_obj = hook(name, func)
        hook_obj.add_condition_func(condition_func)
        setattr(module, func.__name__, hook_obj)


def apply_feature(ops, feature, condition_func=lambda *args, **kwargs: True):
    assert isinstance(ops, (str, list, tuple))
    feature_options = [
        "fallback",
        "autocompare",
        "measure_op_time",
        "dump_op_args",
        "cast_dtype",
        "op_capture",
        "overflow_check",
    ]
    assert feature in feature_options, f"feature must be one of {feature_options}, but got {feature}"
    assert callable(condition_func)
    if feature == "fallback":
        hook_cls = OpFallbackHook
    elif feature == "autocompare":
        hook_cls = OpAutoCompareHook
    elif feature == "measure_op_time":
        hook_cls = OpTimeMeasureHook
    elif feature == "dump_op_args":
        hook_cls = OpObserveHook
    elif feature == "cast_dtype":
        hook_cls = OpDtypeCastHook
    elif feature == "op_capture":
        hook_cls = OpCaptureHook
    elif feature == "overflow_check":
        hook_cls = OpOverflowCheckHook

    if isinstance(ops, str):
        apply_hook_to_ops(ops, hook_cls, condition_func)
    elif isinstance(ops, (list, tuple)):
        for op in ops:
            apply_hook_to_ops(op, hook_cls, condition_func)
    else:
        assert False, f"ops must be str, tuple or list, but got {type(ops)}"
