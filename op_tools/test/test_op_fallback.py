# Copyright (c) 2024, DeepLink.
# 没有fallback的跳过操作
# 也没有针对模型训练设计
import torch
import ditorch
import op_tools
import os


def f():
    a = torch.rand(10, requires_grad=True).cuda().half()
    # assert not a.is_leaf
    a = torch.bernoulli(a) + a + torch.rand_like(a)
    b = a * 2
    c = b + a
    d = c - a
    e = d / c
    sorted, indices = e.sort()  # return torch.return_type.sort
    assert sorted.device.type != "cpu"
    sorted.sum().backward()
    y = torch.cat((a, a, a), dim=0)  # first input type is tuple
    assert not y.is_cpu
    base = 10000
    dim = 1024
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device="cuda", dtype=torch.float32) / dim))
    x = torch.randn(3, 4).cuda().to(torch.float16)  # camb_mlu370 not support bfloat16
    y = x.clone()
    z = y.half()
    n = z.cpu()
    w = torch.Tensor.cpu(n)


f()

# usage 1
with op_tools.OpFallback():
    f()


# usage 2
fallback = op_tools.OpFallback()
fallback.start()
f()
fallback.stop()

f()
print("over")

# usage3
os.environ["OP_FALLBACK_DISABLE_LIST"] = "torch.Tensor.add,torch.Tensor.sub"
fallback.start()
f()
fallback.stop()


# usage4
os.environ["OP_FALLBACK_DISABLE_LIST"] = ""
os.environ["OP_FALLBACK_LIST"] = (
    "torch.bernoulli,torch.Tensor.cpu"  # only capture these op,加了一个需要过滤掉的算子,并保证必备的算子在被选取的范围内
)
fallback.start()
f()
fallback.stop()


# usage5
os.environ["OP_FALLBACK_DISABLE_LIST"] = ""
os.environ["OP_FALLBACK_LIST"] = ""  # 空
fallback.start()
f()
fallback.stop()

# usage6
os.environ["OP_FALLBACK_DISABLE_LIST"] = "torch.Tensor.sort"
os.environ["OP_FALLBACK_LIST"] = "torch.Tensor.sort,torch.Tensor.add,torch.bernoulli"  # 重叠
fallback.start()
f()
fallback.stop()

# usage7
os.environ["OP_FALLBACK_DISABLE_LIST"] = "torch.Tensor.add,torch.Tensor.sub"
if "OP_FALLBACK_LIST" in os.environ:
    del os.environ["OP_FALLBACK_LIST"]  # 删除
fallback.start()
f()
fallback.stop()
