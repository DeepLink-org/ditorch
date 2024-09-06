# Copyright (c) 2024, DeepLink.
import torch
import re
import importlib


def traverse_container(container):
    if isinstance(container, dict):
        for key, value in container.items():
            yield from traverse_container(value)
    elif isinstance(container, (list, tuple, set)):
        for item in container:
            yield from traverse_container(item)
    elif type(container).__module__.startswith("torch.return_types"):
        for i in range(len(container)):
            yield container[i]
    else:
        yield container


def is_cpu_op(*args, **kwargs):
    for obj in traverse_container(args):
        if isinstance(obj, torch.Tensor):
            if not obj.is_cpu:
                return False, obj.device

    return True, "cpu"


def transform_contrainer(obj, func):
    if isinstance(obj, torch.Tensor):
        return func(obj)
    elif isinstance(obj, (tuple, list)):
        return type(obj)([transform_contrainer(v, func) for v in obj])
    elif isinstance(obj, dict):
        return {k: func(v) for k, v in obj.items()}
    elif isinstance(obj, (float, int, complex, str, bool, type(None))):
        return obj
    elif type(obj).__module__.startswith("torch.return_types"):
        return [transform_contrainer(v) for v in obj]
    else:
        return obj


def to_device(device, obj, dtype_cast_dict=dict(), detach=True):
    def func(obj):
        if isinstance(obj, torch.Tensor):
            if obj.dtype in list(dtype_cast_dict.keys()):
                obj = obj.to(dtype_cast_dict[obj.dtype])
            if detach:
                new_obj = obj.detach().to(device)
                new_obj.requires_grad = obj.requires_grad
            else:
                new_obj = obj.to(device)
            return new_obj
        else:
            return obj

    return transform_contrainer(obj, func)


def is_opname_match(name, op_pattern=None):
    """Determine whether the operator matches the template. The template can be a list of operator names or a regular expression."""
    if name is None:
        return False
    if op_pattern is None:
        return True
    op_list = op_pattern.split(",")
    if name in op_list:
        return True

    for pattern in op_list:
        if name in re.findall(pattern, name):
            return True
    return False


def is_inplace_op(name):
    return (
        name.endswith("_")
        and (not name.endswith("__"))
        and (name.startswith("torch.Tensor."))
    )


def get_function_from_string(func_str):
    parts = func_str.split(".")
    attrs = [importlib.import_module(parts[0])]
    for i in range(0, len(parts) - 1):
        attr = getattr(attrs[i], parts[i + 1])
        attrs.append(attr)

    return attrs[len(parts) - 1]


def get_dtype_cast_dict_form_str(config):
    """
    'torch.float16->torch.float32,torch.bfloat16->torch.float32' -> {torch.float16:torch.float32, torch.bfloat16:torch.float32}
    """
    dtype_cast_dict = dict()
    if config is not None:
        for item in config.split(","):
            dtype_cast_dict[get_function_from_string(item.split("->")[0])] = (
                get_function_from_string(item.split("->")[1])
            )
    return dtype_cast_dict
