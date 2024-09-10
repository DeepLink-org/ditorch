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
)


class OpDtypeCastHook(BaseHook):

    def __init__(self, name, func) -> None:
        super().__init__(name, func)
        self.dtype_cast_config_str = os.environ.get(
            "OP_DTYPE_CAST_DICT",
            "torch.float16->torch.float32,torch.bfloat16->torch.float32",
        )
        self.dtype_cast_dict = get_dtype_cast_dict_form_str(self.dtype_cast_config_str)

    def before_call_op(self, *args, **kwargs):
        self.dtype_cast_config_str = os.environ.get(
            "OP_DTYPE_CAST_DICT",
            "torch.float16->torch.float32,torch.bfloat16->torch.float32",
        )
        self.dtype_cast_dict = get_dtype_cast_dict_form_str(self.dtype_cast_config_str)
        with DisableHookGuard():
            self.args_raw = self.args
            self.is_cpu_op, self.device = is_cpu_op(*args, **kwargs)
            if self.is_cpu_op:
                return
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
            self.ins_list = []
            for arg in traverse_container(self.args):
                self.ins_list.append(arg)

            self.raw_ins_list = []
            for arg in traverse_container(self.args_raw):
                self.raw_ins_list.append(arg)

            for i in range(len(self.ins_list)):
                if isinstance(self.ins_list[i], torch.Tensor):
                    if self.ins_list[i].dtype != self.raw_ins_list[i].dtype:
                        print(f"OpDtypeCastHook: {self.name:<50} {i}th arg {self.raw_ins_list[i].dtype} -> {self.ins_list[i].dtype}  config:{self.dtype_cast_config_str}")  # noqa: E501
                        self.dtype_cast_back_dict[self.ins_list[i].dtype] = (
                            self.raw_ins_list[i].dtype
                        )

    def after_call_op(self, result):
        if self.is_cpu_op:
            return
        with DisableHookGuard():
            self.result_raw = result
            self.result = to_device(
                self.device,
                self.result,
                dtype_cast_dict=self.dtype_cast_back_dict,
                detach=False,
            )
            i = -1
            for out in traverse_container(self.result_raw):
                i += 1
                if (
                    isinstance(out, torch.Tensor)
                    and out.dtype in self.dtype_cast_back_dict.keys()
                ):
                    print(f"OpDtypeCastHook: {self.name:<50} {i}th out {out.dtype} -> {self.dtype_cast_back_dict[out.dtype]}  config:{self.dtype_cast_config_str}")  # noqa: E501

    def is_should_apply(self, *args, **kwargs):
        if is_opname_match(self.name, os.getenv("OP_DTYPE_CAST_DISABLE_LIST", "")):
            return False

        return is_opname_match(self.name, os.getenv("OP_DTYPE_CAST_LIST", ".*"))
