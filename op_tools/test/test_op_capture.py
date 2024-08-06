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


# usage1
with op_tools.OpCapture():
    f()

# usage2
capture = op_tools.OpCapture()
capture.start()
for i in range(3):
    f()
capture.stop()
