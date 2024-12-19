# Copyright (c) 2024, DeepLink.

import functools
import torch


def copy_inp(tensor_dest, tensor_src):
    assert tensor_dest.shape == tensor_src.shape, "tensor_dest and tensor_src should have the same shape"
    assert isinstance(tensor_dest, torch.Tensor), "tensor_dest should be a torch.Tensor"
    assert isinstance(tensor_src, torch.Tensor), "tensor_src should be a torch.Tensor"
    if isinstance(tensor_dest, torch.nn.Parameter):
        if isinstance(tensor_src, torch.nn.Parameter):
            tensor_dest.data.copy_(tensor_src.data)
        else:
            tensor_dest.data.copy_(tensor_src)
    else:
        tensor_dest.copy_(tensor_src)


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
