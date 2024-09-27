# Copyright (c) 2024, DeepLink.
# 这里提供了autocompare对一些简单张量操作的的测试，包含两种使用方法
# 可能缺少一些别的基础操作

import torch
import ditorch
import op_tools
import os


def f():
    a = torch.rand(10, requires_grad=True, device="cuda")
    a = torch.bernoulli(a) + a + torch.rand_like(a) + torch.empty_like(a).uniform_() + torch.empty_like(a).normal_()
    b = a * 2 + torch.randperm(a.numel(), dtype=a.dtype, device=a.device).view(a.shape)
    c = b + a
    d = c - a
    e = d / c
    sorted, indices = a.sort()  # return torch.return_type.sort
    sorted.sum().backward()

    x = torch.randn(1).cuda()
    bool(x), x.bool(), x.item()

    m = torch.nn.Linear(4, 5, device="cuda").half()  # cpu not support half
    x = torch.randn(3, 5, 4, device="cuda", requires_grad=True).half()
    y = m(x)
    z = torch.nn.functional.silu(y)
    z.backward(torch.ones_like(z))


f()

# usage1
with op_tools.OpAutoCompare():
    f()

# usage2
capture = op_tools.OpAutoCompare()
capture.start()
for i in range(3):
    f()
capture.stop()


# usage3
os.environ["OP_AUTOCOMPARE_DISABLE_LIST"] = "torch.Tensor.add,torch.Tensor.sub"
capture.start()
f()
capture.stop()

# usage4
os.environ["OP_AUTOCOMPARE_DISABLE_LIST"] = ""
os.environ["OP_AUTOCOMPARE_LIST"] = "torch.Tensor.backward"  # 与EXCLUDE_OPS重复
capture.start()
f()
capture.stop()

# usage5
os.environ["OP_AUTOCOMPARE_DISABLE_LIST"] = ""
os.environ["OP_AUTOCOMPARE_LIST"] = "" # 空
capture.start()
f()
capture.stop()

# usage6
os.environ["OP_AUTOCOMPARE_DISABLE_LIST"] = "torch.Tensor.sort"
os.environ["OP_AUTOCOMPARE_LIST"] = "torch.Tensor.sort,torch.Tensor.add" # 重叠
capture.start()
f()
capture.stop()

# usage7
os.environ["OP_AUTOCOMPARE_DISABLE_LIST"] = "torch.Tensor.add,torch.Tensor.sub"
if "OP_AUTOCOMPARE_LIST" in os.environ:
    del os.environ["OP_AUTOCOMPARE_LIST"] # 删除
capture.start()
f()
capture.stop()

# usage8
os.environ["OP_AUTOCOMPARE_DISABLE_LIST"] = "torch.Tensor.add,torch.Tensor.sub"
os.environ["OP_AUTOCOMPARE_LIST"] = "torch.Tensor.uniform_,torch.empty_like" # 与random_number_gen_ops重叠
capture.start()
f()
capture.stop()


