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
os.environ["OP_CAPTURE_LIST"] = "torch.Tensor.backward"  # capture与EXCLUDE_OPS重复
capture.start()
f()
capture.stop()

# usage5
os.environ["OP_CAPTURE_DISABLE_LIST"] = ""
os.environ["OP_CAPTURE_LIST"] = ""  # 空
capture.start()
f()
capture.stop()

# usage6
os.environ["OP_CAPTURE_DISABLE_LIST"] = "torch.Tensor.sort"
os.environ["OP_CAPTURE_LIST"] = "torch.Tensor.sort,torch.Tensor.add"  # capture和disable重叠
capture.start()
f()
capture.stop()

# usage7
os.environ["OP_CAPTURE_DISABLE_LIST"] = "torch.Tensor.add,torch.Tensor.sub"
if "OP_CAPTURE_LIST" in os.environ:
    del os.environ["OP_CAPTURE_LIST"]  # 删除
capture.start()
f()
capture.stop()
