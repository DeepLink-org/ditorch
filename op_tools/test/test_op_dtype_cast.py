# Copyright (c) 2024, DeepLink.
import torch
import ditorch

import op_tools
import os


def f():
    a = torch.rand(10, 20, requires_grad=True).cuda().half()
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


f()

# usage1
with op_tools.OpDtypeCast():
    f()

# usage2
dtype_caster = op_tools.OpDtypeCast()
dtype_caster.start()
for i in range(3):
    f()
dtype_caster.stop()


# usage3
os.environ["OP_DTYPE_CAST_DISABLE_LIST"] = "torch.Tensor.add,torch.Tensor.sub"
dtype_caster.start()
f()
dtype_caster.stop()


# usage4
os.environ["OP_DTYPE_CAST_DISABLE_LIST"] = ""
os.environ["OP_DTYPE_CAST_LIST"] = "torch.Tensor.sort"  # only cast this op
os.environ["OP_DTYPE_CAST_DICT"] = (
    "torch.half->torch.float32"  # camb 370 not support bfloat16
)
dtype_caster.start()
f()
dtype_caster.stop()
