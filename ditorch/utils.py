# Copyright (c) 2024, DeepLink.

import functools
import torch
from typing import Union


def tensor_op_inp(tensor: Union[torch.Tensor, torch.nn.Parameter],  op_name: str, *op_args, **op_kwargs):
    """
    Perform an inplace operation on a tensor, like tensor.div_(*op_args, **kwargs)

    Args:
        tensor (torch.Tensor, torch.nn.Parameter): The tensor to perform the operation on.
        op_name (str): The name of the operation to perform, must be an inplace operation, such as div_, mul_, add_, sub_, etc.
        op_args (tuple, optional): The arguments to pass to the operation. Defaults to None.
        op_kwargs (dict, optional): The keyword arguments to pass to the operation. Defaults to None.
    """
    assert op_name[-1] == "_", f"{op_name} must be a inplace op"
    assert isinstance(tensor, (torch.Tensor, torch.nn.Parameter)), "tensor_dest should be a torch.Tensor"
    if isinstance(tensor, torch.nn.Parameter):
        func_inp = getattr(tensor.data, op_name)
    else:
        func_inp = getattr(tensor, op_name)
    return func_inp(*op_args, **op_kwargs)


def copy_inp(tensor_dest: Union[torch.Tensor, torch.nn.Parameter], tensor_src: Union[torch.Tensor, torch.nn.Parameter]):
    return tensor_op_inp(tensor_dest, "copy_", tensor_src)


def div_inp(tensor, divisor):
    return tensor_op_inp(tensor, "div_", divisor)


def copy_inp_advanced(tensors_dest, tensors_src):
    if isinstance(tensors_dest, (list, tuple)):
        assert isinstance(tensors_src, (list, tuple)), "the dtype of tensors_src is list or tuple, but tensors_dest is not."
        assert len(tensors_dest) == len(tensors_src), "the len of tensor_dest is not the same as tensor_src."
        return [copy_inp_advanced(tensors_dest[i], tensors_src[i]) for i in range(len(tensors_dest))]
    elif isinstance(tensors_dest, dict):
        assert isinstance(tensors_src, dict), "the dtype of tensors_src is dict, but tensors_dest is not."
        assert len(tensors_dest) == len(tensors_src), "the len of tensors_dest is not the same as tensors_src."
        assert set(tensors_dest.keys()) == set(tensors_src.keys()), "the keys of tensors_dest is not the same as tensors_src."
        return {k: copy_inp_advanced(v, tensors_src[k]) for k, v in tensors_dest.items()}
    elif isinstance(tensors_dest, torch.Tensor):
        assert isinstance(tensors_src, torch.Tensor), "the dtype of tensors_src is not torch.Tensor, but tensors_dest is."
        return copy_inp(tensors_dest, tensors_src)
    else:
        return None


def to_fp32_if_tensor(in_tensors):
    if isinstance(in_tensors, torch.Tensor):
        return in_tensors.to(torch.float32), in_tensors.dtype
    elif isinstance(in_tensors, (list, tuple)):
        return [to_fp32_if_tensor(tensor)[0] for tensor in in_tensors], [to_fp32_if_tensor(tensor)[1] for tensor in in_tensors]
    elif isinstance(in_tensors, dict):
        return {k: to_fp32_if_tensor(v)[0] for k, v in in_tensors.items()},  {k: to_fp32_if_tensor(v)[1] for k, v in in_tensors.items()},
    else:
        return in_tensors, None


def is_to_fp32_tensor(to_fp32: bool):
    def to_fp32_wrapper(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Convert positional arguments to fp32 if possible
            args_fp32, args_old_dtype = to_fp32_if_tensor(args)
            # Convert keyword arguments to fp32 if possible
            kwargs_fp32, kwargs_old_dtype = to_fp32_if_tensor(kwargs)
            ret = func(*args_fp32, **kwargs_fp32)
            copy_inp_advanced(args, args_fp32)
            copy_inp_advanced(kwargs, kwargs_fp32)
            return ret
        if to_fp32:
            print(f"{func.__name__} mocked by fp32.")
            return wrapper
        else:
            return func
    return to_fp32_wrapper
