from abc import ABC, abstractmethod
import torch
import ditorch
from .utils import to_device, is_cpu_op, get_function_from_string


import argparse


class OpRunnerHook(ABC):
    def before_forward(self):
        pass

    def after_forward(self):
        pass

    def before_backward(self):
        pass

    def after_backward(self):
        pass


class OpRunner:
    def __init__(self, dir=".", hook=OpRunnerHook()) -> None:
        self.dir = dir
        self.hook = hook
        self.hook.runner = self

    def load_forward_input(self):
        self.input = torch.load(self.dir + "/input.pth", map_location="cpu")
        self.args_cpu = self.input["args"]
        self.kwargs_cpu = self.input["kwargs"]
        self.args = to_device("cuda", self.args_cpu)
        self.kwargs = to_device("cuda", self.kwargs_cpu or {})
        self.name = self.input["name"]
        self.fun = get_function_from_string(self.name)

    def load_forward_output(self):
        self.output = torch.load(self.dir + "/output.pth", map_location="cpu")
        self.output = to_device("cuda", self.output)

    def load_backward_data(self):
        self.grad_inputs = torch.load(self.dir + "/grad_inputs.pth", map_location="cpu")
        self.grad_outputs_cpu = torch.load(
            self.dir + "/grad_outputs.pth", map_location="cpu"
        )
        self.grad_outputs = to_device("cuda", self.grad_outputs_cpu)

    def run_forward(self):
        self.load_forward_input()
        self.hook.before_forward()
        self.result = self.fun(*self.args, **self.kwargs)
        self.hook.after_forward()

    def run_backward(self):
        self.load_backward_data()
        self.hook.before_backward()
        self.result.backward(self.grad_outputs)
        self.hook.after_backward()
