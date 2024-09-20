# Copyright (c) 2024, DeepLink.
import torch
import gc
import os
import time
import atexit

from .base_hook import BaseHook, DisableHookGuard
from .utils import (
    to_device,
    is_cpu_op,
    traverse_container,
    is_inplace_op,
    is_view_op,
    is_opname_match,
    compare_result,
    is_random_number_gen_op,
)
from .save_op_args import save_op_args, serialize_args_to_dict

from .pretty_print import pretty_print_op_args, dict_data_list_to_table


SKIP_LIST_OPS = [
    "torch.nn.functional.embedding",
]


class AutoCompareResultCache:
    global_autocompare_result = []

    def __init__(self) -> None:
        self.file_name = f"/share/work/dongkaixing/logs/op_tools/op_tools_results/op_autocompare_result/op_autocompare_info_pid{os.getpid()}_{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}.csv"  # noqa: E501
        self.dir = self.file_name[0 : self.file_name.rfind("/")]

    def append(self, forward_id, compare_info):
        for result in compare_info["result_list"]:
            self.global_autocompare_result.append({"forward_id": forward_id, **result})

        if len(self.global_autocompare_result) > int(os.getenv("OP_TOOLS_MAX_CACHE_SIZE", "1000")):
            self.write_to_file()

    def write_to_file(self):
        if len(self.global_autocompare_result) == 0:
            return
        table = dict_data_list_to_table(self.global_autocompare_result)
        print(table)
        self.global_autocompare_result.clear()
        data_string = table.get_csv_string()

        os.makedirs(self.dir, exist_ok=True)
        with open(self.file_name, "a+") as f:
            f.write(data_string)
            f.close
        print(f"op autocompare result saved to {self.file_name}")


compare_result_cache = AutoCompareResultCache()


def dump_all_autocompare_info():
    compare_result_cache.write_to_file()


class BackwardHookHandle:
    def __init__(self, compare_hook) -> None:
        self.compare_hook = compare_hook

    def register_grad_fn_hook(self, tensor):
        hook_handle = None

        def grad_fun(grad_inputs, grad_outputs):
            self.compare_hook.run_backward_on_cpu(grad_inputs, grad_outputs)
            self.compare_hook.compare_all_grad()
            hook_handle.remove()

        hook_handle = tensor.grad_fn.register_hook(grad_fun)
        return grad_fun


class OpAutoCompareHook(BaseHook):
    AUTO_COMPARE_DTYPE_CAST_DICT = {
        torch.half: torch.float32,
        torch.bfloat16: torch.float32,
    }

    def __init__(self, name, func) -> None:
        super().__init__(name, func)

    def copy_input_to_cpu(self):
        self.args_cpu = to_device(
            "cpu",
            self.args,
            detach=True,
        )
        self.kwargs_cpu = to_device(
            "cpu",
            self.kwargs or {},
            detach=True,
        )

    def run_forward_on_cpu(self):
        try:
            self.result_cpu = self.func(*self.args_cpu, **self.kwargs_cpu)
            self.result_device = to_device("cpu", self.result, detach=True)
            self.dtype_cast_dict = dict()
            args_cpu = self.args_cpu
        except Exception as e:  # noqa: F841
            self.dtype_cast_dict = OpAutoCompareHook.AUTO_COMPARE_DTYPE_CAST_DICT
            # some op on cpu backend not support half, bfloat16
            self.args_cpu = to_device(
                "cpu",
                self.args_cpu,
                dtype_cast_dict=self.dtype_cast_dict,
                detach=True,
            )
            self.kwargs_cpu = to_device(
                "cpu",
                self.kwargs_cpu or {},
                dtype_cast_dict=self.dtype_cast_dict,
                detach=True,
            )
            # RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
            if (is_inplace_op(self.name) or is_view_op(self.name)) and self.args[0].requires_grad:
                args_cpu = [item for item in self.args_cpu]
                args_cpu[0] = args_cpu[0].clone()
                self.result_cpu = self.func(*args_cpu, **self.kwargs_cpu)
            else:
                args_cpu = self.args_cpu
                self.result_cpu = self.func(*self.args_cpu, **self.kwargs_cpu)

            self.result_device = to_device(
                "cpu",
                self.result,
                dtype_cast_dict=self.dtype_cast_dict,
                detach=True,
            )

    def run_backward_on_cpu(self, grad_inputs, grad_output):
        self.grad_outputs_cpu = to_device("cpu", grad_output, dtype_cast_dict=self.dtype_cast_dict, detach=True)
        self.grad_inputs_cpu = to_device("cpu", grad_inputs, dtype_cast_dict=self.dtype_cast_dict, detach=True)
        for arg_cpu in traverse_container(self.args_cpu):
            if isinstance(arg_cpu, torch.Tensor) and arg_cpu.grad is not None:
                arg_cpu.grad.zero_()

        self.args_cpu_grad = []

        def post_hook(grad_inputs, grad_outputs):
            self.args_cpu_grad = [grad_input for grad_input in grad_inputs]

        for result_cpu in traverse_container(self.result_cpu):
            if isinstance(result_cpu, torch.Tensor) and result_cpu.requires_grad:
                handle = result_cpu.grad_fn.register_hook(post_hook)
                result_cpu.backward(*self.grad_outputs_cpu)
                handle.remove()

    def register_backward_hook_for_grads(self):
        self.backward_hook_handle = BackwardHookHandle(self)
        for result in traverse_container(self.result):
            if isinstance(result, torch.Tensor):
                if result.grad_fn is not None:
                    self.backward_hook_handle.register_grad_fn_hook(result)

    def compare_forward_result(self):
        compare_info = compare_result(self.name, self.result_device, self.result_cpu)
        compare_result_cache.append(self.forward_op_id, compare_info)

        allclose = compare_info["allclose"]
        self.forward_allclose = allclose
        if not allclose:
            pretty_print_op_args(
                self.name,
                serialize_args_to_dict(*self.args, **self.kwargs),
                serialize_args_to_dict(self.result),
            )
            self.save_forward_args()

    def count_params_with_requires_grad(self):
        count = 0
        for arg in traverse_container(self.args):
            if isinstance(arg, torch.Tensor) and arg.requires_grad:
                count = count + 1
        return count

    def compare_all_grad(self):
        self.args_grad = self.grad_inputs_cpu
        compare_info = compare_result(self.name + " grad", self.args_cpu_grad, self.args_grad)

        compare_result_cache.append(self.forward_op_id, compare_info)

        if not compare_info["allclose"]:
            # Parameters are not saved when forward accuracy is normal
            if self.forward_allclose:
                self.save_forward_args()
            self.save_backward_args()
        self = None
        gc.collect()

    def save_forward_args(self):
        save_op_args(
            self.name,
            f"{self.identifier}/device/input",
            *self.args,
            **self.kwargs,
        )
        save_op_args(self.name, f"{self.identifier}/device/output", self.result)
        save_op_args(
            self.name,
            f"{self.identifier}/cpu/input",
            *self.args_cpu,
            **self.kwargs_cpu,
        )
        save_op_args(self.name, f"{self.identifier}/cpu/output", self.result_cpu)

    def save_backward_args(self):
        save_op_args(
            self.name,
            f"{self.identifier}/device/grad_outputs",
            *tuple(self.grad_outputs_cpu),
        )
        save_op_args(
            self.name,
            f"{self.identifier}/device/grad_inputs",
            *tuple(self.args_grad),
        )
        save_op_args(
            self.name,
            f"{self.identifier}/cpu/grad_inputs",
            *tuple(self.args_cpu_grad),
        )
        save_op_args(
            self.name,
            f"{self.identifier}/cpu/grad_outputs",
            *tuple(self.grad_outputs_cpu),
        )

    def before_call_op(self, *args, **kwargs):
        self.forward_op_id = self.id
        self.identifier = f"autocompare/{self.id}/{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}"
        with DisableHookGuard():
            self.is_cpu_op, self.device = is_cpu_op(*args, **kwargs)
            if self.is_cpu_op:
                return
            self.copy_input_to_cpu()

    def after_call_op(self, result):  # noqa:C901
        if self.is_cpu_op:
            return
        with DisableHookGuard():
            self.run_forward_on_cpu()

            if self.result is None and self.result_cpu is None:
                print(f"{self.name} output is None, no check for accuracy")
                return

            self.compare_forward_result()

            self.register_backward_hook_for_grads()

            self.args = to_device("cpu", self.args, detach=True)
            self.kwargs = to_device("cpu", self.kwargs or {}, detach=True)

    def is_should_apply(self, *args, **kwargs):
        if is_random_number_gen_op(self.name):
            return False

        if self.name in SKIP_LIST_OPS:
            return False

        if self.name.startswith("torch.empty"):
            return False

        if is_opname_match(self.name, os.getenv("OP_AUTOCOMPARE_DISABLE_LIST", "")):
            return False

        return is_opname_match(self.name, os.getenv("OP_AUTOCOMPARE_LIST", ".*"))


atexit.register(dump_all_autocompare_info)
