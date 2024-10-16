# Copyright (c) 2024, DeepLink.
import os
from .utils import is_opname_match, traverse_container, is_inf_or_nan, garbage_collect, compute_tensor_features
from .base_hook import BaseHook, DisableHookGuard
import torch

from .save_op_args import serialize_args_to_dict
from .pretty_print import packect_data_to_dict_list, dict_data_list_to_table


class BackwardHookHandle:
    def __init__(self, name, id, location) -> None:
        self.name = name
        self.id = id
        self.location = location

    def process(self, grad_inputs, grad_outputs):
        index = -1
        info_list = []
        for arg in grad_inputs + grad_outputs:
            index = index + 1
            item_name = f"grad_inputs[{index}]" if index < len(grad_inputs) else f"grad_outputs[{index - len(grad_inputs)}]"
            if isinstance(arg, torch.Tensor):
                info = {"name": self.name + " " + item_name}
                info.update(compute_tensor_features(arg))
                info_list.append(info)
        print("\n" * 2)
        print(f"{self.name}     forward_id: {self.id}")
        print(f"{self.location}")
        grad_output_list = packect_data_to_dict_list(self.name + " grad_output", serialize_args_to_dict(*grad_outputs))
        grad_inputs_list = packect_data_to_dict_list(self.name + " grad_inputs", serialize_args_to_dict(*grad_inputs))
        print(dict_data_list_to_table(grad_output_list + grad_inputs_list))
        print(dict_data_list_to_table(info_list))
        self = None
        garbage_collect()

    def register_grad_fn_hook(self, tensor):
        hook_handle = None

        def grad_fun(grad_inputs, grad_outputs):
            hook_handle.remove()
            with torch.no_grad():
                with DisableHookGuard():
                    self.process(grad_inputs, grad_outputs)

        hook_handle = tensor.grad_fn.register_hook(grad_fun)
        return grad_fun


class OpOverflowCheckHook(BaseHook):
    def __init__(self, name, func) -> None:
        super().__init__(name, func)

    def packect_data_to_dict(self):
        info_list = []
        ins = []
        outs = []
        for arg in traverse_container(self.args):
            ins.append(arg)
        for out in traverse_container(self.result):
            outs.append(out)

        index = -1
        for arg in ins + outs:
            index = index + 1
            item_name = f"input[{index}]" if index < len(ins) else f"output[{index - len(ins)}]"
            info = {}
            info["name"] = self.name + " " + item_name
            if isinstance(arg, torch.Tensor):
                info.update(compute_tensor_features(arg))
            else:
                info["inf_or_nan"] = is_inf_or_nan(arg)

            info_list.append(info)

        return info_list

    def dump_op_args(self):
        info_list = self.packect_data_to_dict()
        inputs_list = packect_data_to_dict_list(self.name + " inputs", serialize_args_to_dict(*self.args, **self.kwargs))
        output_list = packect_data_to_dict_list(self.name + " outputs", serialize_args_to_dict(self.result))
        forward_args_table = dict_data_list_to_table(inputs_list + output_list)
        print("\n" * 2)
        print(f"{self.name}    forward_id: {self.id}")
        print(f"{self.current_location}")
        print(forward_args_table)
        print(dict_data_list_to_table(info_list))
        print("\n" * 2)

    def register_backward_hook_for_grads(self):
        self.backward_hook_handle = BackwardHookHandle(self.name, self.id, self.current_location)

        for result in traverse_container(self.result):
            if isinstance(result, torch.Tensor):
                if result.grad_fn is not None:
                    self.backward_hook_handle.register_grad_fn_hook(result)

    def before_call_op(self, *args, **kwargs):
        pass

    def after_call_op(self, result):
        with DisableHookGuard():
            with torch.no_grad():
                self.dump_op_args()
            self.register_backward_hook_for_grads()
            garbage_collect()

    def is_should_apply(self, *args, **kwargs):
        if is_opname_match(self.name, os.getenv("OP_OVERFLOW_CHECK_DISABLE_LIST", "")):
            return False
        return is_opname_match(self.name, os.getenv("OP_OVERFLOW_CHECK_LIST", ".*"))
