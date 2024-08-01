import torch
import ditorch

import op_tools


def f():
    a = torch.rand(10, requires_grad=True).cuda()
    b = a * 2
    b.sum()


f()

with op_tools.OpFallback():
    f()


fallback = op_tools.OpFallback()
fallback.start()
f()
fallback.stop()

f()
print("over")
