from .base_hook import BaseHook, DisableHookGuard

from .save_op_args import save_op_args


class OpCaptureHook(BaseHook):
    def __init__(self, name) -> None:
        super().__init__(name)

    def before_call_op(self, *args, **kwargs):
        with DisableHookGuard():
            id = f"{self.id}/input"
            save_op_args(self.name, id, args, kwargs)

    def after_call_op(self, result):
        with DisableHookGuard():
            id = f"{self.id}/output"
            save_op_args(self.name, id, self.result)
