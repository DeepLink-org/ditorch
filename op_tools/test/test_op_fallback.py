import torch
import ditorch

import op_tools


def f():
    a = torch.rand(10, requires_grad=True).cuda().half()
    a = torch.bernoulli(a) + a + torch.rand_like(a)
    b = a * 2
    c = b + a
    d = c - a
    e = d / c
    sorted, indices = e.sort()  # return torch.return_type.sort
    assert not sorted.is_cpu
    sorted.sum().backward()
    print(c)
    y = torch.cat((a, a, a), dim=0)  # first input type is tuple
    assert not y.is_cpu


f()

# usage 1
with op_tools.OpFallback():
    f()


##usage 2
fallback = op_tools.OpFallback()
fallback.start()
f()
fallback.stop()

f()
print("over")
