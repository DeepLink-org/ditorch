import torch
import ditorch

import op_tools


def f():
    a = torch.rand(10, requires_grad=True, device="cuda").half()
    a = torch.bernoulli(a) + a + torch.rand_like(a)

    b = a * 2
    c = b + a
    d = c - a
    e = d / c
    sorted, indices = a.sort()  # return torch.return_type.sort
    sorted.sum().backward()


f()

# usage1
print("normal run, op dispatch process:")
with op_tools.OpAutoCompare():
    f()

print("\n")

# usage2
comparer = op_tools.OpAutoCompare()
comparer.start()
f()
comparer.stop()
