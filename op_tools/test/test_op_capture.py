# Copyright (c) 2024, DeepLink.
import torch
import ditorch

import op_tools
import os


def f():
    a = torch.rand(10, 20, requires_grad=True).cuda()
    b = a * 2
    c = b + a
    d = c - a
    e = d / c
    sorted, indices = e.sort()  # return torch.return_type.sort
    y = sorted[2:8:2, ::3]
    y.sum().backward()

    m = torch.nn.Linear(4, 4, device="cuda").half()
    x = torch.randn(3, 4, 4, device="cuda", requires_grad=True, dtype=torch.half)
    y = m(x)
    y.backward(torch.ones_like(y))


# usage1
with op_tools.OpCapture():
    f()

# usage2
capture = op_tools.OpCapture()
capture.start()
for i in range(3):
    f()
capture.stop()


# usage3
os.environ["OP_CAPTURE_DISABLE_LIST"] = "torch.Tensor.add,torch.Tensor.sub"
capture.start()
f()
capture.stop()


# usage4
os.environ["OP_CAPTURE_DISABLE_LIST"] = ""
os.environ["OP_CAPTURE_LIST"] = "torch.Tensor.sort"  # only capture these op
capture.start()
f()
capture.stop()
