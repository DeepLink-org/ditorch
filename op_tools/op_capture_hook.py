# Copyright (c) 2024, DeepLink.
import os
import torch
import time
from .base_hook import BaseHook, DisableHookGuard
from .utils import traverse_container, is_opname_match, garbage_collect
from .save_op_args import save_op_args, serialize_args_to_dict
from .pretty_print import dict_data_list_to_table, packect_data_to_dict_list


class BackwardHookHandle:
    def __init__(self, name, id) -> None:
        self.name = name
        self.id = id

    def register_grad_fun_hook(self, tensor):
        hook_handle = None

        def grad_fun(grad_inputs, grad_outputs):
            hook_handle.remove()

            save_op_args(self.name, f"{self.id}/grad_inputs", *tuple(grad_inputs))
            save_op_args(self.name, f"{self.id}/grad_outputs", *tuple(grad_outputs))

            grad_output_list = packect_data_to_dict_list(self.name + " grad_output", serialize_args_to_dict(*grad_outputs))
            grad_inputs_list = packect_data_to_dict_list(self.name + " grad_inputs", serialize_args_to_dict(*grad_inputs))
            backward_args_table = dict_data_list_to_table(grad_output_list + grad_inputs_list)
            print(f"{self.name} forward_id:{id}\n{backward_args_table}", "\n" * 4)

        hook_handle = tensor.grad_fn.register_hook(grad_fun)

        return grad_fun


class OpCaptureHook(BaseHook):
    def __init__(self, name, func) -> None:
        super().__init__(name, func)

    def op_forward_args_to_table(self):
        inputs_list = packect_data_to_dict_list(self.name + " inputs", serialize_args_to_dict(*self.args, **self.kwargs))
        output_list = packect_data_to_dict_list(self.name + " outputs", serialize_args_to_dict(self.result))
        forward_args_table = dict_data_list_to_table(inputs_list + output_list)
        return forward_args_table

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

            table = self.op_forward_args_to_table()
            print(f"{self.name}    forward_id:{self.forward_op_id}    {self.current_location} \n{table}", "\n"*4)

            self.backward_hook_handle = BackwardHookHandle(self.name, self.forward_op_id)

            for result in traverse_container(self.result):
                if isinstance(result, torch.Tensor):
                    if result.grad_fn is not None:
                        self.backward_hook_handle.register_grad_fun_hook(result)

            self = None
            garbage_collect()

    def is_should_apply(self, *args, **kwargs):
        if is_opname_match(self.name, os.getenv("OP_CAPTURE_DISABLE_LIST", "")):
            return False

        if not is_opname_match(self.name, os.getenv("OP_CAPTURE_LIST", ".*")):
            return False

        return True
