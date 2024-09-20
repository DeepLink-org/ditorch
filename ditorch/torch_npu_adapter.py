# Copyright (c) 2024, DeepLink.
import torch
import torch_npu  # noqa: F401
from torch_npu.contrib import transfer_to_npu  # noqa: F401


def current_stream(device=None):
    old_device = torch.cuda.current_device()
    if device is None:
        device = old_device
    torch.cuda.set_device(device)
    stream = torch_npu.npu.current_stream(device)
    torch.cuda.set_device(old_device)
    return stream


torch.cuda.current_stream = current_stream


def get_device_capability():
    return (8, 0)


torch.cuda.get_device_capability = get_device_capability

torch._C._cuda_setStream = torch_npu._C._npu_setStream

torch._C._cuda_setDevice = torch_npu._C._npu_setDevice


def _set_cached_tensors_enabled(enable):
    print("_set_cached_tensors_enabled not supported in torch_npu")


torch._C._set_cached_tensors_enabled = _set_cached_tensors_enabled


def _add_cached_tensor(tensor):
    print("_add_cached_tensor not supported in torch_npu")
    raise NotImplementedError


torch._C._add_cached_tensor = _add_cached_tensor
