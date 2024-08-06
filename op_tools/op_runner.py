import torch
import ditorch
from .utils import to_device, is_cpu_op, get_function_from_string


import argparse


class OpRunner:
    def __init__(self, dir=".") -> None:
        self.dir = dir

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
        self.grad_outputs = torch.load(
            self.dir + "/grad_outputs.pth", map_location="cpu"
        )

    def run_forward(self):
        self.load_forward_input()
