# Copyright (c) 2024, DeepLink.
import torch
import math
import gc
import os

from .base_hook import BaseHook, DisableHookGuard
from .utils import (
    to_device,
    is_cpu_op,
    traverse_container,
    is_inplace_op,
    get_dtype_cast_dict_form_str,
)
from .op_fallback_hook import OpFallbackHook
from .save_op_args import save_op_args, serialize_args_to_dict


class OpDtypeCastHook(BaseHook):
    def __init__(self, name) -> None:
        super().__init__(name)
        self.dtype_cast_config_str = os.environ.get(
            "OP_DTYPE_CAST_DICT",
            "torch.float16->torch.float32,torch.bfloat16->torch.float32",
        )
        self.dtype_cast_dict = get_dtype_cast_dict_form_str(self.dtype_cast_config_str)

    def before_call_op(self, *args, **kwargs):
        with DisableHookGuard():
            self.args_raw = self.args
            self.is_cpu_op, self.device = is_cpu_op(*args, **kwargs)
            if self.is_cpu_op:
                return
            self.args = to_device(self.device, self.args, self.dtype_cast_dict, False)
            self.kwargs = to_device(
                self.device, self.kwargs or {}, self.dtype_cast_dict, False
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
                        print(
                            f"OpDtypeCastHook: {self.name:<50} {i}th arg {self.raw_ins_list[i].dtype} -> {self.ins_list[i].dtype}  config:{self.dtype_cast_config_str}"
                        )
                        self.dtype_cast_back_dict[
                            self.ins_list[i].dtype
                        ] = self.raw_ins_list[i].dtype

    def after_call_op(self, result):
        if self.is_cpu_op:
            return
        with DisableHookGuard():
            self.result_raw = result
            self.result = to_device(
                self.device, self.result, self.dtype_cast_back_dict, False
            )
            i = -1
            for out in traverse_container(self.result_raw):
                i += 1
                if (
                    isinstance(out, torch.Tensor)
                    and out.dtype in self.dtype_cast_back_dict.keys()
                ):
                    print(
                        f"OpDtypeCastHook: {self.name:<50} {i}th out {out.dtype} -> {self.dtype_cast_back_dict[out.dtype]}  config:{self.dtype_cast_config_str}"
                    )
