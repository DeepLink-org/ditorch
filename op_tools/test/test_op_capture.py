import torch
import ditorch

import op_tools


def f():
    a = torch.rand(10, requires_grad=True).cuda()
    b = a * 2
    b.sum().backward()
    print(a)


# usage1
with op_tools.OpCapture():
    f()

# usage2
tool = op_tools.OpCapture()
tool.start()
for i in range(3):
    f()
tool.stop()
