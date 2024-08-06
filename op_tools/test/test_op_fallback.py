import torch
import ditorch

import op_tools


def f():
    a = torch.rand(10, requires_grad=True).cuda()
    b = a * 2
    c = b + a
    d = c - a
    e = d / c
    sorted, indices = a.sort()  # return torch.return_type.sort
    sorted.sum().backward()
    print(c)


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
