# Copyright (c) 2024, DeepLink.
import os
import torch
import time
from .base_hook import BaseHook, DisableHookGuard
from .utils import traverse_container, is_opname_match
from .save_op_args import save_op_args


class BackwardHookHandle:
    def __init__(self, name, id) -> None:
        self.name = name
        self.id = id

    def grad_fun_hook(self):
        def grad_fun(grad_inputs, grad_outputs):
            save_op_args(self.name, f"{self.id}/grad_inputs", *tuple(grad_inputs))
            save_op_args(self.name, f"{self.id}/grad_outputs", *tuple(grad_outputs))

        return grad_fun


class OpCaptureHook(BaseHook):
    def __init__(self, name, func) -> None:
        super().__init__(name, func)

    def before_call_op(self, *args, **kwargs):
        self.forward_op_id = f"{self.id}/{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}"
        with DisableHookGuard():
            name = self.name

            id = f"{self.forward_op_id}/input"

            save_op_args(name, id, *args, **kwargs)

    def after_call_op(self, result):

        with DisableHookGuard():
            id = f"{self.forward_op_id}/output"
            save_op_args(self.name, id, self.result)

            self.backward_hook_handle = BackwardHookHandle(self.name, self.forward_op_id)

            for result in traverse_container(self.result):
                if isinstance(result, torch.Tensor):
                    if result.grad_fn is not None:
                        result.grad_fn.register_hook(self.backward_hook_handle.grad_fun_hook())

    def is_should_apply(self, *args, **kwargs):
        if is_opname_match(self.name, os.getenv("OP_CAPTURE_DISABLE_LIST", "")):
            return False

        if not is_opname_match(self.name, os.getenv("OP_CAPTURE_LIST", ".*")):
            return False

        return True
