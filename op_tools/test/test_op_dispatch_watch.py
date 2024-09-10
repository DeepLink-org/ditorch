# Copyright (c) 2024, DeepLink.
import torch
from torch.utils._python_dispatch import TorchDispatchMode
import ditorch

import op_tools


def f():
    a = torch.rand(10, requires_grad=True).cuda()
    b = a + a
    c = b - b
    d = c * 2
    e = torch.sin(d)
    f = torch.cos(e)
    g = f.abs()
    g.sum()
    g.backward(torch.ones_like(g))

    x = torch.randn(3, 4, requires_grad=True).cuda()
    assert x.requires_grad

    y = x.to(torch.half)
    assert y.requires_grad


f()

with op_tools.OpFallback():
    f()

# usage1
print("normal run, op dispatch process:")
with op_tools.OpDispatchWatcher():
    f()

print("\n")

# usage2
watcher = op_tools.OpDispatchWatcher()
watcher.start()
f()
watcher.stop()


print("\n" * 2)
print("dispatch process of the operator when it is fallbacked:")
with op_tools.OpDispatchWatcher():
    with op_tools.OpFallback():
        f()


print("\n" * 2)
print("dispatch process of the operator when autocompare is enabled:")
with op_tools.OpDispatchWatcher():
    with op_tools.OpAutoCompare():
        f()


class TesTorchDispatchMode(TorchDispatchMode):

    def __torch_dispatch__(self, func, types, args, kwargs=None):
        return func(*args, **kwargs)


x = torch.randn(3, 4, requires_grad=True, device="cuda")
x_cpu = x.cpu()
n = x.half()
with TesTorchDispatchMode():
    y = x.half()
    z = x + x
    y_cpu = x_cpu + x_cpu
assert y_cpu.requires_grad
assert n.requires_grad
assert z.requires_grad

# When TorchDispatchMode is turned on, the requires_grad of the output Tensor of
# some operators (aten.to.dtype, aten._to_copy.default, aten.to.dtype_layout, aten.view.default eq.)
# will be accidentally changed to False. This should be a bug in pytorch.
# I once tried to modify the requires_grad attribute of the output Tensor,
# but found that I could not modify this attribute directly in __torch_dispatch__

# assert (y.requires_grad)
