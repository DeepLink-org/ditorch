# Copyright (c) 2024, DeepLink.

import functools
import torch


def to_fp32_if_tensor(in_tensors):
    if isinstance(in_tensors, torch.Tensor):
        return in_tensors.to(torch.float32)
    elif isinstance(in_tensors, (list, tuple)):
        return [to_fp32_if_tensor(tensor) for tensor in in_tensors]
    elif isinstance(in_tensors, dict):
        return {k: to_fp32_if_tensor(v) for k, v in in_tensors.items()}
    else:
        return in_tensors


def is_to_fp32_tensor(to_fp32: bool):
    def to_fp32_wrapper(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Convert positional arguments to fp32 if possible
            args_fp32 = to_fp32_if_tensor(args)
            # Convert keyword arguments to fp32 if possible
            kwargs_fp32 = to_fp32_if_tensor(kwargs)
            return func(*args_fp32, **kwargs_fp32)
        if to_fp32:
            print(f"{func.__name__} mocked by fp32.")
            return wrapper
        else:
            return func
    return to_fp32_wrapper
