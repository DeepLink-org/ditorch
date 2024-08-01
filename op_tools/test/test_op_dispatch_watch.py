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

print("normal run")
with op_tools.OpDispatchWatcher():
    f()
print("\n" * 3)
print("fallback cpu")
with op_tools.OpDispatchWatcher():
    with op_tools.OpFallback():
        f()


watcher = op_tools.OpDispatchWatcher()
watcher.start()
f()
watcher.stop()
