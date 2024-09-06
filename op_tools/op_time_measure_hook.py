# Copyright (c) 2024, DeepLink.
import os
import torch
import ditorch
import time
from .base_hook import BaseHook, DisableHookGuard

from .save_op_args import serialize_args_to_dict
from .utils import is_opname_match


class BackwardHookHandle:
    def __init__(self, name, id) -> None:
        self.name = name
        self.id = id

    def grad_fun_prehook(self):
        def grad_fun(grad_inputs):
            torch.cuda.current_stream().synchronize()
            self.start_time = time.time()

        return grad_fun

    def grad_fun_posthook(self):
        def grad_fun(grad_inputs, grad_outputs):
            torch.cuda.current_stream().synchronize()
            self.end_time = time.time()
            self.backward_elasped = self.end_time - self.start_time
            print(
                f"OpTimeMeasureHook: {self.name:<30} backward elasped: {(self.backward_elasped * 1000):>10.8f} ms     grad_inputs: {serialize_args_to_dict(grad_inputs)} output: {serialize_args_to_dict(grad_outputs)}"
            )

        return grad_fun


class OpTimeMeasureHook(BaseHook):
    def __init__(self, name, func) -> None:
        super().__init__(name, func)

    def before_call_op(self, *args, **kwargs):
        torch.cuda.current_stream().synchronize()
        self.start_time = time.time()

    def after_call_op(self, result):
        torch.cuda.current_stream().synchronize()
        self.end_time = time.time()
        self.foward_elasped = self.end_time - self.start_time

        self.backward_hook_handle = BackwardHookHandle(self.name, self.id)
        if isinstance(self.result, torch.Tensor):
            if self.result.grad_fn is not None:
                self.result.grad_fn.register_hook(
                    self.backward_hook_handle.grad_fun_posthook()
                )
                self.result.grad_fn.register_prehook(
                    self.backward_hook_handle.grad_fun_prehook()
                )
        elif isinstance(self.result, (tuple, list)) or type(
            self.result
        ).__module__.startswith("torch.return_types"):
            # torch.return_types is a structseq, aka a "namedtuple"-like thing defined by the Python C-API.
            for i in range(len(self.result)):
                if (
                    isinstance(self.result[i], torch.Tensor)
                    and self.result[i].grad_fn is not None
                ):
                    self.result[i].grad_fn.register_hook(
                        self.backward_hook_handle.grad_fun_posthook()
                    )

                    self.result[i].grad_fn.register_prehook(
                        self.backward_hook_handle.grad_fun_prehook()
                    )

        with DisableHookGuard():
            print(
                f"OpTimeMeasureHook: {self.name:<30} forward elasped:  {(self.foward_elasped * 1000):>10.8f} ms     input: {serialize_args_to_dict(*self.args, **self.kwargs)} output: {serialize_args_to_dict(self.result)}"
            )

    def is_should_apply(self, *args, **kwargs):
        if is_opname_match(self.name, os.getenv("OP_TIME_MEASURE_DISABLE_LIST", "")):
            return False

        return is_opname_match(self.name, os.getenv("OP_TIME_MEASURE_LIST", ".*"))
