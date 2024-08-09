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
    else:
        yield container


def is_cpu_op(*args, **kwargs):
    for obj in traverse_container(args):
        if isinstance(obj, torch.Tensor):
            if not obj.is_cpu:
                return False, obj.device

    return True, "cpu"


def to_device(device, obj, dtype_cast_dict=dict()):
    if isinstance(obj, torch.Tensor):
        if obj.dtype in list(dtype_cast_dict.keys()):
            obj = obj.to(dtype_cast_dict[obj.dtype], non_blocking=False)
        return obj.to(device, non_blocking=False)
    elif isinstance(obj, (tuple, list)):
        return type(obj)([to_device(device, v, dtype_cast_dict) for v in obj])
    elif isinstance(obj, dict):
        return {k: to_device(device, v, dtype_cast_dict) for k, v in obj.items()}
    elif isinstance(obj, (float, int, complex, str, bool, type(None))):
        return obj
    elif type(obj).__module__.startswith("torch.return_types"):
        return [to_device(device, v, dtype_cast_dict) for v in obj]
    else:
        return obj


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


def get_function_from_string(func_str):
    parts = func_str.split(".")
    attrs = [importlib.import_module(parts[0])]
    for i in range(0, len(parts) - 1):
        attr = getattr(attrs[i], parts[i + 1])
        attrs.append(attr)

    return attrs[len(parts) - 1]
