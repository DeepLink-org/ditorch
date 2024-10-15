# Copyright (c) 2024, DeepLink.
import os
from .utils import is_opname_match
from .base_hook import BaseHook, DisableHookGuard

from .save_op_args import serialize_args_to_dict
from .pretty_print import packect_data_to_dict_list, dict_data_list_to_table


class OpObserveHook(BaseHook):
    def __init__(self, name, func) -> None:
        super().__init__(name, func)

    def before_call_op(self, *args, **kwargs):
        pass

    def after_call_op(self, result):
        with DisableHookGuard():
            inputs_list = packect_data_to_dict_list(self.name + " inputs", serialize_args_to_dict(*self.args, **self.kwargs))
            output_list = packect_data_to_dict_list(self.name + " outputs", serialize_args_to_dict(self.result))
            forward_args_table = dict_data_list_to_table(inputs_list + output_list)
            print("\n" * 2)
            print(f"{self.name}  {self.id}")
            print(f"{self.current_location}")
            print(forward_args_table)
            print("\n" * 2)

    def is_should_apply(self, *args, **kwargs):
        if is_opname_match(self.name, os.getenv("OP_OBSERVE_DISABLE_LIST", "")):
            return False
        return is_opname_match(self.name, os.getenv("OP_OBSERVE_LIST", ".*"))
