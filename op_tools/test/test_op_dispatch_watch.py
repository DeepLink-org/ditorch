import torch
import ditorch

import op_tools


def f():
    a = torch.rand(10, requires_grad=True).cuda()
    b = a + a
    c = b - b
    d = c * 2
    e = torch.sin(d)
    f = torch.cos(e)
    g = e.abs()
    f.sum()


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
