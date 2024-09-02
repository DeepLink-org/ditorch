# Copyright (c) 2024, DeepLink.
from .base_hook import BaseHook, DisableHookGuard

from .save_op_args import serialize_args_to_dict


class OpDispatchWatcherHook(BaseHook):
    def __init__(self, name) -> None:
        super().__init__(name)

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
