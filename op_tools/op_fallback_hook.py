# Copyright (c) 2024, DeepLink.
import torch

from .base_hook import BaseHook, DisableHookGuard
from .utils import to_device, is_cpu_op, is_opname_match, garbage_collect, get_option
from .save_op_args import serialize_args_to_dict
from .pretty_print import packect_data_to_dict_list, dict_data_list_to_table


class OpFallbackHook(BaseHook):
    FALLBACK_DTYPE_CAST_DICT = {
        torch.half: torch.float32,
        torch.bfloat16: torch.float32,
    }

    def __init__(self, name, func) -> None:
        super().__init__(name, func)

    def get_dtype_convert_back_dict(self):
        convert_dict = dict()
        for i in range(len(self.args)):
            arg = self.args[i]
            arg_raw = self.args_device[i]
            if isinstance(arg, torch.Tensor):
                if arg.dtype != arg_raw.dtype:
                    convert_dict[arg.dtype] = arg_raw.dtype
        for k in self.kwargs.keys():
            arg = self.kwargs[k]
            arg_raw = self.kwargs_device[k]
            if isinstance(arg, torch.Tensor):
                if arg.dtype != arg_raw.dtype:
                    convert_dict[arg.dtype] = arg_raw.dtype
        self.dtype_convert_back_dict = convert_dict
        return convert_dict

    def before_call_op(self, *args, **kwargs):
        with DisableHookGuard():
            self.dtype_cast_dict = dict()
            self.is_cpu_op, self.device = is_cpu_op(*args, **kwargs)
            if self.is_cpu_op:
                return
            self.args_device = self.args
            self.kwargs_device = self.kwargs
            self.args = to_device(
                "cpu",
                self.args,
            )
            self.kwargs = to_device(
                "cpu",
                self.kwargs or {},
            )

    def after_call_op(self, result):
        if self.is_cpu_op:
            return
        with DisableHookGuard():
            if self.result is not None and self.exception is None:
                self.result_cpu = self.result
                self.dtype_convert_back_dict = dict()
            else:
                # cpu backend do not support half or bfloat16
                self.dtype_cast_dict = OpFallbackHook.FALLBACK_DTYPE_CAST_DICT
                self.args = to_device(
                    "cpu",
                    self.args_device,
                    dtype_cast_dict=self.dtype_cast_dict,
                )
                self.kwargs = to_device(
                    "cpu",
                    self.kwargs_device or {},
                    dtype_cast_dict=self.dtype_cast_dict,
                )
                try:
                    self.result_cpu = self.func(*self.args, **self.kwargs)
                except Exception as e:
                    self.exception = e
                else:
                    self.exception = None

                self.dtype_convert_back_dict = self.get_dtype_convert_back_dict()

            self.result = to_device(self.device, self.result_cpu, self.dtype_convert_back_dict)
            self.dump_op_args()
            self = None
            garbage_collect()

    def dump_op_args(self):
        data_dict_list = []
        data_dict_list += packect_data_to_dict_list(
            self.name + " input(device)",
            serialize_args_to_dict(*self.args_device, **self.kwargs_device)
        )
        data_dict_list += packect_data_to_dict_list(
            self.name + " input(cpu)",
            serialize_args_to_dict(*self.args, **self.kwargs),
        )
        data_dict_list += packect_data_to_dict_list(self.name + " output(device)", serialize_args_to_dict(self.result))
        data_dict_list += packect_data_to_dict_list(self.name + " output(cpu)", serialize_args_to_dict(self.result_cpu))

        table = dict_data_list_to_table(data_dict_list)
        dtype_cast_info = ""
        if len(self.dtype_cast_dict) > 0:
            dtype_cast_info = "cpu_dtype_cast_info: " + str(self.dtype_cast_dict)

        print("\n" * 2)
        print(f"fallback    {self.name}    forward_id: {self.id}  {dtype_cast_info}")
        print(f"{self.current_location}")
        print(table)
        print("\n" * 2)

    def is_should_apply(self, *args, **kwargs):
        BLACK_OP_LIST = ["torch.Tensor.cpu"]
        if self.name in BLACK_OP_LIST:
            return False

        if is_opname_match(self.name, get_option("OP_FALLBACK_DISABLE_LIST", "")):
            return False
        # if name in VIEW_OPS:
        #    return False

        return is_opname_match(self.name, get_option("OP_FALLBACK_LIST", ".*"))
