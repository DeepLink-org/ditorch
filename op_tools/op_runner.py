from abc import ABC
import os
import torch
import ditorch
import time
from .utils import to_device, is_cpu_op, get_function_from_string, traverse_container
import argparse
from .op_autocompare_hook import compare_result
from .save_op_args import serialize_args_to_dict


class OpRunnerHook(ABC):
    def before_forward(self):
        pass

    def after_forward(self):
        pass

    def before_backward(self):
        pass

    def after_backward(self):
        pass


class AsyncEventTimer(OpRunnerHook):
    def __init__(self) -> None:
        super().__init__()
        self.forward_start_event = torch.cuda.Event(
            enable_timing=True, blocking=False, interprocess=False
        )
        self.forward_end_event = torch.cuda.Event(
            enable_timing=True, blocking=False, interprocess=False
        )

        self.backward_start_event = torch.cuda.Event(
            enable_timing=True, blocking=False, interprocess=False
        )
        self.backward_end_event = torch.cuda.Event(
            enable_timing=True, blocking=False, interprocess=False
        )

    def before_forward(self):
        self.forward_start_event.record(torch.cuda.current_stream)

    def after_forward(self):
        self.forward_end_event.record(torch.cuda.current_stream)

    def before_backward(self):
        self.backward_start_event.record(torch.cuda.current_stream)

    def after_backward(self):
        self.backward_end_event.record(torch.cuda.current_stream)


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


class OpAccyChecker(OpRunnerHook):
    def __init__(self) -> None:
        super().__init__()

    def before_forward(self):
        pass

    def after_forward(self):
        self.runner.result_cpu = self.runner.fun(
            *self.runner.args_cpu, **self.runner.kwargs_cpu
        )
        allclose, max_diff = compare_result(
            self.runner.name, self.runner.result, self.runner.result_cpu
        )
        if not allclose and max_diff > 1e-3:
            print(
                f"OpAccyChecker: {self.name:<50} input: {serialize_args_to_dict(*self.args, **self.kwargs)}"
            )
            print(
                f"OpAccyChecker: {self.name:<50} output: {serialize_args_to_dict(self.result)['args']}"
            )

    def before_backward(self):
        pass

    def after_backward(self):
        self.runner.run_backward_on_cpu()
        for i in range(len(self.runner.args)):
            arg = self.runner.args[i]
            arg_cpu = self.runner.args_cpu[i]
            if isinstance(arg, torch.Tensor) and arg.requires_grad:
                allclose, max_diff = compare_result(
                    self.runner.name + f" {i}th input grad ",
                    self.runner.grad_inputs["args"][i],
                    arg_cpu.grad,
                )
                if not allclose and max_diff > 1e-3:
                    print(f"{i}th grad is not allclose ")


class OpRunner:
    def __init__(self, dir=".", hook=OpRunnerHook()) -> None:
        self.dir = dir
        self.hooks = []
        self.add_hook(hook)
        print(f"{dir}")
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
        self.fun = get_function_from_string(self.name)

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

    def run_backward_on_cpu(self):
        self.result_cpu = self.fun(*self.args_cpu, **self.kwargs_cpu)
        for arg_cpu in traverse_container(self.args_cpu):
            if isinstance(arg_cpu, torch.Tensor) and arg_cpu.grad is not None:
                arg_cpu.grad.zero_()
        self.result_cpu.backward(
            *self.grad_outputs_cpu["args"], **self.grad_outputs_cpu["kwargs"]
        )

    def run_forward(self):
        self.run_before_forward()
        self.result = self.fun(*self.args, **self.kwargs)
        self.run_after_forward()

    def run_backward(self):
        if self.grad_outputs is None:
            return
        self.run_before_backward()
        self.result.backward(*self.grad_outputs["args"], **self.grad_outputs["kwargs"])
        self.run_after_backward()
