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
    return (7, 5)


torch.cuda.get_device_capability = get_device_capability
