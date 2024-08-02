import torch


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
        return {k: to_device(device, v) for k, v in obj}
    else:
        return obj
