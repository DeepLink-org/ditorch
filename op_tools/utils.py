import torch
import re


def is_cpu_op(*args, **kwargs):
    device = "cpu"
    for v in args:
        if isinstance(v, torch.Tensor):
            if v.is_cpu:
                return True, "cpu"
            else:
                device = v.device
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            if v.is_cpu:
                return True, "cpu"
            else:
                device = v.device
    return False, device


def to_device(device, obj):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, (tuple, list)):
        return type(obj)([to_device(device, v) for v in obj])
    elif isinstance(obj, dict):
        return {k: to_device(device, v) for k, v in obj.items()}
    elif isinstance(obj, (float, int, complex, str, bool, type(None))):
        return obj
    elif type(obj).__module__.startswith("torch.return_types"):
        return [to_device(device, v) for v in obj]
    else:
        print(f"{__file__} unhandled type {obj}")
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