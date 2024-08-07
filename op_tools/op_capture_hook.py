from .base_hook import BaseHook, DisableHookGuard

from .save_op_args import save_op_args
import torch


class BackwardHookHandle:
    def __init__(self, name, id) -> None:
        self.name = name
        self.id = id

    def grad_fun_hook(self):
        def grad_fun(grad_inputs, grad_outputs):
            save_op_args(self.name, f"{self.id}/grad_inputs", *tuple(grad_inputs))
            save_op_args(self.name, f"{self.id}/grad_outputs", *tuple(grad_outputs))

        return grad_fun


class OpCaptureHook(BaseHook):
    def __init__(self, name) -> None:
        super().__init__(name)

    def before_call_op(self, *args, **kwargs):
        with DisableHookGuard():
            name = self.name

            id = f"{self.id}/input"

            save_op_args(name, id, *args, **kwargs)

    def after_call_op(self, result):

        with DisableHookGuard():
            id = f"{self.id}/output"
            save_op_args(self.name, id, self.result)

            self.backward_hook_handle = BackwardHookHandle(self.name, self.id)
            if isinstance(self.result, torch.Tensor):
                if self.result.grad_fn is not None:
                    self.result.grad_fn.register_hook(
                        self.backward_hook_handle.grad_fun_hook()
                    )
            elif isinstance(self.result, (tuple, list)) or type(
                self.result
            ).__module__.startswith("torch.return_types"):
                # torch.return_types is a structseq, aka a "namedtuple"-like thing defined by the Python C-API.
                for i in range(len(self.result)):
                    if (
                        isinstance(self.result[i], torch.Tensor)
                        and self.result[i].grad_fn is not None
                    ):
                        self.result[i].grad_fn.register_hook(
                            self.backward_hook_handle.grad_fun_hook()
                        )
