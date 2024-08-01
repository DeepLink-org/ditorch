import torch

import torch_npu
from torch_npu.contrib import transfer_to_npu

import op_capture


def f():
    a = torch.rand(10, requires_grad=True).cuda()
    b = a * 2
    b.sum()


f()

with op_capture.OpFallback():
    f()


fallback = op_capture.OpFallback()
fallback.start()
f()
fallback.stop()
