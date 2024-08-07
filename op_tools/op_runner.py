from abc import ABC
import os
import torch
import ditorch
import time
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


class EventTimer(OpRunnerHook):
    def __init__(self) -> None:
        super().__init__()
        self.event_pair_list = list()

    def before_forward(self):
        start_event = torch.cuda.Event(
            enable_timing=True, blocking=False, interprocess=False
        )
        end_event = torch.cuda.Event(
            enable_timing=True, blocking=False, interprocess=False
        )


class SyncExecuteTimer(OpRunnerHook):
    def __init__(self) -> None:
        super().__init__()

    def before_forward(self):
        torch.cuda.current_stream().synchronize()
        self.forward_start_time = time.time()

    def after_forward(self):
        torch.cuda.current_stream().synchronize()
        self.forward_end_time = time.time()
        self.elasped_time = self.forward_end_time - self.forward_start_time
        print(
            f"SyncExecuteTimer: {self.runner.name} forward elasped {self.elasped_time * 1000:>.8f} ms "
        )

    def before_backward(self):
        torch.cuda.current_stream().synchronize()
        self.backward_start_time = time.time()

    def after_backward(self):
        torch.cuda.current_stream().synchronize()
        self.backward_end_time = time.time()
        self.elasped_time = self.backward_end_time - self.forward_start_time
        print(
            f"SyncExecuteTimer: {self.runner.name} backward elasped {self.elasped_time * 1000:>.8f} ms"
        )


class OpRunner:
    def __init__(self, dir=".", hook=OpRunnerHook()) -> None:
        self.dir = dir
        self.hook = hook
        self.hook.runner = self
        print(f"{dir}")

    def load_forward_input(self):
        self.input = torch.load(self.dir + "/input.pth", map_location="cpu")
        self.args_cpu = self.input["args"]
        self.kwargs_cpu = self.input["kwargs"]
        self.args = to_device("cuda", self.args_cpu)
        self.kwargs = to_device("cuda", self.kwargs_cpu or {})
        self.name = self.input["name"]
        # self.id = self.input["id"]
        self.fun = get_function_from_string(self.name)

    def load_forward_output(self):
        self.output = torch.load(self.dir + "/output.pth", map_location="cpu")
        self.output = to_device("cuda", self.output)

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
        self.load_forward_input()
        self.hook.before_forward()
        self.result = self.fun(*self.args, **self.kwargs)
        self.hook.after_forward()

    def run_backward(self):
        self.load_backward_data()
        if self.grad_outputs is None:
            return
        self.hook.before_backward()
        self.result.backward(*self.grad_outputs["args"], **self.grad_outputs["kwargs"])
        self.hook.after_backward()
