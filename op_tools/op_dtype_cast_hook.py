# Copyright (c) 2024, DeepLink.
import torch
import os

from .base_hook import BaseHook, DisableHookGuard
from .utils import (
    to_device,
    is_cpu_op,
    traverse_container,
    get_dtype_cast_dict_form_str,
    is_opname_match,
    is_view_op,
    is_dtype_cast_op,
    garbage_collect,
    get_option,
)
from .pretty_print import dict_data_list_to_table


class OpDtypeCastHook(BaseHook):

    def __init__(self, name, func) -> None:
        super().__init__(name, func)

    def before_call_op(self, *args, **kwargs):
        self.dtype_cast_config_str = os.environ.get(
            "OP_DTYPE_CAST_DICT",
            "torch.float16->torch.float32,torch.bfloat16->torch.float32",
        )
        self.dtype_cast_dict = get_dtype_cast_dict_form_str(self.dtype_cast_config_str)
        with DisableHookGuard():
            self.is_cpu_op, self.device = is_cpu_op(*args, **kwargs)
            if self.is_cpu_op:
                return

            self.raw_ins_dtype_list = []
            for arg in traverse_container(self.args):
                if isinstance(arg, torch.Tensor):
                    self.raw_ins_dtype_list.append(arg.dtype)
                else:
                    self.raw_ins_dtype_list.append(None)

            for dtype in set(self.dtype_cast_dict.keys()):
                if dtype not in self.raw_ins_dtype_list:
                    self.dtype_cast_dict.pop(dtype)

            self.args = to_device(
                self.device,
                self.args,
                dtype_cast_dict=self.dtype_cast_dict,
                detach=False,
            )
            self.kwargs = to_device(
                self.device,
                self.kwargs or {},
                dtype_cast_dict=self.dtype_cast_dict,
                detach=False,
            )
            self.dtype_cast_back_dict = {}
            self.ins_dtype_list = []
            for arg in traverse_container(self.args):
                if isinstance(arg, torch.Tensor):
                    self.ins_dtype_list.append(arg.dtype)
                else:
                    self.ins_dtype_list.append(None)

            self.data_dict_list = []
            for i in range(len(self.ins_dtype_list)):
                if self.ins_dtype_list[i] != self.raw_ins_dtype_list[i]:
                    self.dtype_cast_back_dict[self.ins_dtype_list[i]] = self.raw_ins_dtype_list[i]
                    data_dict = {
                        "name": self.name,
                        "target": f"input[{i}]",
                        "action": f"{self.raw_ins_dtype_list[i]} -> {self.ins_dtype_list[i]}",
                        "config": self.dtype_cast_config_str,
                    }
                    self.data_dict_list.append(data_dict)

    def after_call_op(self, result):
        if self.is_cpu_op:
            return
        with DisableHookGuard():
            self.raw_result_dtype_list = []
            for arg in traverse_container(self.result):
                if isinstance(arg, torch.Tensor):
                    self.raw_result_dtype_list.append(arg.dtype)
                else:
                    self.raw_result_dtype_list.append(None)

            self.result = to_device(
                self.device,
                self.result,
                dtype_cast_dict=self.dtype_cast_back_dict,
                detach=False,
            )

            self.result_dtype_list = []
            for arg in traverse_container(self.result):
                if isinstance(arg, torch.Tensor):
                    self.result_dtype_list.append(arg.dtype)
                else:
                    self.result_dtype_list.append(None)

            i = -1
            for out in traverse_container(self.raw_result_dtype_list):
                i += 1
                if out in self.dtype_cast_back_dict.keys():
                    data_dict = {
                        "name": self.name,
                        "target": f"output[{i}]",
                        "action": f"{out} -> {self.dtype_cast_back_dict[out]}",
                        "config": self.dtype_cast_config_str,
                    }
                    self.data_dict_list.append(data_dict)
            if len(self.data_dict_list) > 0:
                print("\n" * 2, f"cast_dtype    {self.name}   forward_id: {self.id}")
                print(f"{self.current_location}")
                print(dict_data_list_to_table(self.data_dict_list))
                print("\n" * 2)
            result = self.result
            self = None
            garbage_collect()
            return result

    def is_should_apply(self, *args, **kwargs):
        if is_opname_match(self.name, get_option("OP_DTYPE_CAST_DISABLE_LIST", "")):
            return False

        if is_view_op(self.name):
            return False

        if is_dtype_cast_op(self.name, *args, **kwargs):
            return False

        return is_opname_match(self.name, get_option("OP_DTYPE_CAST_LIST", ".*"))
