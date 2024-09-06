# Copyright (c) 2024, DeepLink.
import unittest

import torch
import ditorch
import torch.nn as nn
import torch.nn.functional as F

import psutil
import os
import op_tools
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square conv kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x = x.view(-1, int(x.nelement() / x.shape[0]))  # todo: fix error on camb: RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
        x = x.reshape(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TestOpTools(unittest.TestCase):

    def test_func(self):
        model = LeNet().to(device=device)
        input = torch.randn(32, 1, 32, 32, requires_grad=True).to(device=device)
        target = torch.randint(0, 10, (32,)).to(device=device)
        output = model(input)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        optimizer.zero_grad()
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

    def test_op_capture(self):
        with op_tools.OpCapture():
            self.test_func()

    def test_op_autocompare(self):
        with op_tools.OpAutoCompare():
            self.test_func()

    def test_dump_op_args(self):
        with op_tools.OpDispatchWatcher():
            self.test_func()

    def test_op_autocompare_memusage(self):
        process = psutil.Process(os.getpid())

        torch.cuda.synchronize()
        gc.collect()
        host_memory1 = process.memory_info().rss
        run_time = 50  # The more times you run it, the better it will reflect the problem, but too many will waste CI resources.
        for i in range(run_time):
            self.test_op_autocompare()

        torch.cuda.synchronize()
        gc.collect()
        host_memory2 = process.memory_info().rss
        host_memusage = host_memory2 - host_memory1
        print(f"Memory usage {host_memusage >> 10} KB")
        assert host_memusage / run_time <= (
            10 << 20
        ), f"Memory usage {int(host_memusage / run_time) >> 20}MB should not increase after running the test {host_memory1} , {host_memory2}"

    def test_op_autocompare_inplace_op_and_requires_grad(self):
        with op_tools.OpAutoCompare():
            x = torch.randn(32, 1, 32, 32, requires_grad=True).to(device=device)
            y = x * 2
            z = y.div_(2)
            z.backward(torch.ones_like(z))

    def test_op_autocompare_mul_op(self):
        with op_tools.OpAutoCompare():
            x = torch.randn(32, 1, 32, 32, requires_grad=True).to(device=device)
            z = torch.mul(x, x)
            z.backward(torch.ones_like(z))

    def test_op_autocompare_copy_op(self):
        with op_tools.OpAutoCompare():
            x = torch.randn(32, 1, 32, 32, requires_grad=True)
            assert x.device == torch.device("cpu")
            y = x.to(device=device)
            assert y.is_cpu == (device.type == "cpu"), f"{y.device} {device}"
            z = torch.add(x, x)
            assert z.is_cpu == (device.type != "cpu"), f"{z.device} {device}"
            z = torch.add(y, y)
            assert z.is_cpu == (device.type == "cpu"), f"{z.device} {device}"
            z.backward(torch.ones_like(z))

            e = z.cpu()
            assert e.is_cpu

    def test_op_dtype_cast(self):
        with op_tools.OpDtypeCast():
            input = torch.ones(
                (5, 5), dtype=torch.bfloat16, device="cuda"
            ).requires_grad_()
            weight = torch.ones(
                (5, 5), dtype=torch.bfloat16, device="cuda"
            ).requires_grad_()
            output = torch.nn.functional.linear(input, weight)
            label = torch.ones_like(output)
            output.backward(label)
            assert input.grad is not None and input.grad.dtype == torch.bfloat16
            assert weight.grad is not None and weight.grad.dtype == torch.bfloat16


if __name__ == "__main__":
    unittest.main()
