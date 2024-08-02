import torch
import ditorch

import op_tools


def f():
    a = torch.rand(10, requires_grad=True).cuda()
    b = a * 2
    b.sum()


f()

# usage1
print("normal run, op dispatch process:")
with op_tools.OpAutocompare():
    f()

print("\n")

# usage2
comparer = op_tools.OpAutocompare()
comparer.start()
f()
comparer.stop()
