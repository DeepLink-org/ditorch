# Copyright (c) 2024, DeepLink.
from abc import ABC
import os
import torch
from .utils import to_device, get_function_from_string, traverse_container, is_inplace_op
from .op_autocompare_hook import OpAutoCompareHook
from .op_time_measure_hook import OpTimeMeasureHook


class OpRunnerHook(ABC):
    def before_forward(self):
        pass

    def after_forward(self):
        pass

    def before_backward(self):
        pass

    def after_backward(self):
        pass


class SyncExecuteTimer(OpRunnerHook):
    def __init__(self) -> None:
        super().__init__()
        self.time_measure_hook = None
        self.raw_func = None

    def before_forward(self):
        if self.time_measure_hook is None:
            self.raw_func = self.runner.func
            self.time_measure_hook = OpTimeMeasureHook(self.runner.name, self.runner.func)
            self.runner.func = self.time_measure_hook
        else:
            self.runner.func = self.time_measure_hook

    def after_forward(self):
        self.runner.func = self.raw_func


class OpAccyChecker(OpRunnerHook):
    def __init__(self) -> None:
        super().__init__()
        self.raw_func = None
        self.aucompare_hook = None

    def before_forward(self):
        if self.aucompare_hook is None:
            self.aucompare_hook = OpAutoCompareHook(self.runner.name, self.runner.func)
            self.runner.func = self.aucompare_hook
        else:
            self.runner.func = self.aucompare_hook

    def after_forward(self):
        self.runner.func = self.raw_func


class OpRunner:
    def __init__(self, dir=".", hook=OpRunnerHook()) -> None:
        self.dir = dir
        self.hooks = []
        self.add_hook(hook)
        print(f"dir: {dir}")
        self.load_forward_input()
        self.load_forward_output()
        self.load_backward_data()

    def add_hook(self, hook):
        if hook is not None and isinstance(hook, OpRunnerHook):
            hook.runner = self
            self.hooks.append(hook)

    def run_before_forward(self):
        for hook in self.hooks:
            hook.before_forward()

    def run_after_forward(self):
        for hook in self.hooks:
            hook.after_forward()

    def run_before_backward(self):
        for hook in self.hooks:
            hook.before_backward()

    def run_after_backward(self):
        for hook in self.hooks:
            hook.after_backward()

    def load_forward_input(self):
        self.input = torch.load(self.dir + "/input.pth", map_location="cpu")
        self.args_cpu = self.input["args"]
        self.kwargs_cpu = self.input["kwargs"] or {}
        self.args = to_device("cuda", self.args_cpu)
        self.kwargs = to_device("cuda", self.kwargs_cpu)
        self.name = self.input["name"]
        self.func = get_function_from_string(self.name)

    def load_forward_output(self):
        self.output_cpu = torch.load(self.dir + "/output.pth", map_location="cpu")
        self.output = to_device("cuda", self.output_cpu)

    def load_backward_data(self):
        grad_inputs_path = self.dir + "/grad_inputs.pth"
        if os.path.exists(grad_inputs_path):
            self.grad_inputs_cpu = torch.load(grad_inputs_path, map_location="cpu")
            self.grad_inputs = to_device("cuda", self.grad_inputs_cpu)
        else:
            self.grad_inputs = None
            self.grad_inputs_cpu = None
        grad_outputs_path = self.dir + "/grad_outputs.pth"
        if os.path.exists(grad_outputs_path):
            self.grad_outputs_cpu = torch.load(grad_outputs_path, map_location="cpu")
            self.grad_outputs = to_device("cuda", self.grad_outputs_cpu)
        else:
            self.grad_outputs = None
            self.grad_outputs_cpu = None

    def run_forward(self):
        if is_inplace_op(self.name):
            self.args = to_device("cuda", self.args_cpu)
            self.kwargs = to_device("cuda", self.kwargs_cpu)
        self.run_before_forward()
        self.result = self.func(*self.args, **self.kwargs)
        self.run_after_forward()

    def run_backward(self):
        if self.grad_outputs is None:
            return
        self.run_before_backward()
        for result in traverse_container(self.result):
            if isinstance(result, torch.Tensor) and result.requires_grad:
                result.backward(*self.grad_outputs["args"], **self.grad_outputs["kwargs"])
        self.run_after_backward()
