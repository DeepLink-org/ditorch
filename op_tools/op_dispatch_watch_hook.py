# Copyright (c) 2024, DeepLink.
import os
from .utils import is_opname_match
from .base_hook import BaseHook, DisableHookGuard

from .save_op_args import serialize_args_to_dict


class OpDispatchWatcherHook(BaseHook):
    def __init__(self, name, func) -> None:
        super().__init__(name, func)

    def before_call_op(self, *args, **kwargs):
        with DisableHookGuard():
            print(
                f"OpDispatchWatcherHook: {self.name} input: {serialize_args_to_dict(*args, **kwargs)}"
            )

    def after_call_op(self, result):
        with DisableHookGuard():
            print(
                f"OpDispatchWatcherHook: {self.name} output: {serialize_args_to_dict(self.result)}"
            )

    def is_should_apply(self, *args, **kwargs):
        if is_opname_match(self.name, os.getenv("OP_DISPATCH_WATCH_DISABLE_LIST", "")):
            return False
        return is_opname_match(self.name, os.getenv("OP_DISPATCH_WATCH_LIST", ".*"))
