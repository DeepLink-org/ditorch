import torch
import torch_npu

from torch_npu.contrib import transfer_to_npu

import op_capture


def f():
    a = torch.rand(10, requires_grad=True).cuda()
    b = a * 2
    b.sum()


f()

with op_capture.OpFallback():
    f()

print("normal run")
with op_capture.OpDispatchWatcher():
    f()
print("\n" * 3)
print("fallback cpu")
with op_capture.OpDispatchWatcher():
    with op_capture.OpFallback():
        f()


watcher = op_capture.OpDispatchWatcher()
watcher.start()
f()
watcher.stop()
