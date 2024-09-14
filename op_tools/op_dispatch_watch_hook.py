# Copyright (c) 2024, DeepLink.
import os
from .utils import is_opname_match
from .base_hook import BaseHook, DisableHookGuard

from .save_op_args import serialize_args_to_dict
from .pretty_print import pretty_print_op_args


class OpDispatchWatcherHook(BaseHook):
    def __init__(self, name, func) -> None:
        super().__init__(name, func)

    def before_call_op(self, *args, **kwargs):
        pass

    def after_call_op(self, result):
        with DisableHookGuard():
            pretty_print_op_args(
                self.name,
                serialize_args_to_dict(*self.args, **self.kwargs),
                serialize_args_to_dict(self.result),
            )

    def is_should_apply(self, *args, **kwargs):
        if is_opname_match(self.name, os.getenv("OP_DISPATCH_WATCH_DISABLE_LIST", "")):
            return False
        return is_opname_match(self.name, os.getenv("OP_DISPATCH_WATCH_LIST", ".*"))
