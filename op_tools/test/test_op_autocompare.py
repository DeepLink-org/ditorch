# Copyright (c) 2024, DeepLink.
import torch
import ditorch

import op_tools


def f():
    a = torch.rand(10, requires_grad=True, device="cuda").half()
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
comparer = op_tools.OpAutoCompare()
comparer.start()
f()
comparer.stop()
