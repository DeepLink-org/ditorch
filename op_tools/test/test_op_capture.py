import torch

import torch_npu
from torch_npu.contrib import transfer_to_npu

import op_capture


def f():
    a = torch.rand(10, requires_grad=True).cuda()
    b = a * 2
    b.sum().backward()


with op_capture.OpCapture():
    f()

tool = op_capture.OpCapture()

tool.start()
for i in range(3):
    f()
tool.stop()
