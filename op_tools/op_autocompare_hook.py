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
)
from .save_op_args import save_op_args, serialize_args_to_dict

from .pretty_print import pretty_print_op_args, dict_data_list_to_table

RANDOM_NUMBER_GEN_OPS = [
    "torch.Tensor.random_",
    "torch.Tensor.uniform_",
    "torch.Tensor.normal_",
    "torch.Tensor.bernoulli_",
    "torch.Tensor.poisson_",
    "torch.Tensor.multinomial_",
    "torch.Tensor.random",
    "torch.Tensor.uniform",
    "torch.Tensor.normal",
    "torch.Tensor.bernoulli",
    "torch.Tensor.poisson",
    "torch.Tensor.multinomial",
    "torch.rand",
    "torch.randperm",
    "torch.bernoulli",
    "torch.poisson",
    "torch.randint_like",
    "torch.randint",
    "torch.randn",
    "torch.randn_like",
    "torch.multinomial",
    "torch.nn.init.kaiming_uniform",
    "torch.nn.init.kaiming_uniform_",
    "torch.nn.init.trunc_normal_",
    "torch.nn.init.uniform",
    "torch.nn.init.normal",
    "torch.nn.init.uniform_",
    "torch.nn.init.normal_",
    "torch.nn.init.warnings",
    "torch.nn.init.xavier_normal",
    "torch.nn.init.xavier_normal_",
    "torch.nn.init.xavier_uniform",
    "torch.nn.init.kaiming_normal",
    "torch.nn.init.xavier_uniform_",
    "torch.nn.init.kaiming_normal_",
]

SKIP_LIST_OPS = []


class BackwardHookHandle:
    def __init__(self, compare_hook) -> None:
        self.compare_hook = compare_hook

    def register_grad_fn_hook(self, tensor):
        hook_handle = None

        def grad_fun(grad_inputs, grad_outputs):
            hook_handle.remove()
            self.compare_hook.run_backward_on_cpu(grad_inputs, grad_outputs)
            self.compare_hook.compare_all_grad()

        hook_handle = tensor.grad_fn.register_hook(grad_fun)
        return grad_fun

    def register_tensor_hook(self, index, tensor):
        hook_handle = None

        def grad_fun(grad):
            hook_handle.remove()
            self.compare_hook.set_input_grad(index, grad)
            self.compare_hook.compare_all_grad()

        hook_handle = tensor.register_hook(grad_fun)

        return grad_fun


global_autocompare_result = []


class OpAutoCompareHook(BaseHook):
    AUTO_COMPARE_DTYPE_CAST_DICT = {
        torch.half: torch.float32,
        torch.bfloat16: torch.float32,
    }

    def __init__(self, name, func) -> None:
        super().__init__(name, func)

    def before_call_op(self, *args, **kwargs):
        self.forward_op_id = self.id
        self.identifier = f"autocompare/{self.id}/{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}"
        with DisableHookGuard():
            self.is_cpu_op, self.device = is_cpu_op(*args, **kwargs)
            if self.is_cpu_op:
                return
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

    def after_call_op(self, result):  # noqa:C901
        if self.is_cpu_op:
            return
        with DisableHookGuard():
            self.result = result
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

            if is_inplace_op(self.name):
                compare_info = compare_result(self.name, self.args[0], args_cpu[0])
                allclose = compare_info["allclose"]
                if not allclose:
                    self.save_forward_args()
                compare_info.update({"forward_id": self.forward_op_id})
                global_autocompare_result.append(compare_info)

            compare_info = compare_result(self.name, self.result_device, self.result_cpu)
            compare_info.update({"forward_id": self.forward_op_id})
            global_autocompare_result.append(compare_info)
            allclose = compare_info["allclose"]

            if self.result is None:
                print(f"{self.name} output is None, acc not checked")
                return

            self.forward_allclose = allclose
            if not allclose:
                pretty_print_op_args(
                    self.name,
                    serialize_args_to_dict(*self.args, **self.kwargs),
                    serialize_args_to_dict(self.result),
                )
                self.save_forward_args()

            self.backward_hook_handle = BackwardHookHandle(self)
            for result in traverse_container(self.result):
                if isinstance(result, torch.Tensor):
                    if result.grad_fn is not None:
                        self.backward_hook_handle.register_grad_fn_hook(result)
            index = -1
            for arg in traverse_container(self.args):
                index += 1
                if isinstance(arg, torch.Tensor) and arg.requires_grad:
                    self.backward_hook_handle.register_tensor_hook(index, arg)

            self.args = to_device("cpu", self.args, detach=True)
            self.kwargs = to_device("cpu", self.kwargs or {}, detach=True)

    def run_backward_on_cpu(self, grad_inputs, grad_output):
        self.grad_outputs_cpu = to_device("cpu", grad_output, dtype_cast_dict=self.dtype_cast_dict, detach=True)
        self.grad_inputs_cpu = to_device("cpu", grad_inputs, dtype_cast_dict=self.dtype_cast_dict, detach=True)
        for arg_cpu in traverse_container(self.args_cpu):
            if isinstance(arg_cpu, torch.Tensor) and arg_cpu.grad is not None:
                arg_cpu.grad.zero_()

        for result_cpu in traverse_container(self.result_cpu):
            if isinstance(result_cpu, torch.Tensor) and result_cpu.requires_grad:
                result_cpu.backward(*self.grad_outputs_cpu)

        self.args_cpu_grad = []
        for i in range(len(self.args_cpu)):
            if isinstance(self.args_cpu[i], torch.Tensor) and self.args_cpu[i].grad is not None:
                self.args_cpu_grad.append(self.args_cpu[i].grad)
            else:
                self.args_cpu_grad.append(None)

    def count_params_with_requires_grad(self):
        count = 0
        for arg in traverse_container(self.args):
            if isinstance(arg, torch.Tensor) and arg.requires_grad:
                count = count + 1
        return count

    def compare_all_grad(self):
        """
        Since the order in which the two backward hooks registered by different operators \
        are called is not exactly the same, and the gradient can only be calculated on the CPU \
        after the grad_fn hook is called, there is a judgment here, \
        because only when the gradient on the CPU is calculated \
        and all the gradients of the device parameters are obtained, \
        can the gradient information be compared.
        """
        if not hasattr(self, "args_cpu_grad"):
            return
        if not hasattr(self, "args_grad"):
            return

        # Check if all gradients have been obtained
        for i in range(len(self.args)):
            arg = self.args[i]
            if isinstance(arg, torch.Tensor) and (arg.requires_grad and self.args_grad[i] is None):
                return

        compare_info = compare_result(self.name + " grad", self.args_cpu_grad, self.args_grad)
        compare_info.update({"forward_id": self.forward_op_id})
        global_autocompare_result.append(compare_info)
        if not compare_info["allclose"]:
            # Parameters are not saved when forward accuracy is normal
            if self.forward_allclose:
                self.save_forward_args()
            self.save_backward_args()
        self = None
        gc.collect()

    def set_input_grad(self, index, grad):
        if not hasattr(self, "args_grad"):
            self.args_grad = [None for i in range(len(self.args))]
        self.args_grad[index] = to_device("cpu", grad, dtype_cast_dict=self.dtype_cast_dict, detach=True)

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

    def is_should_apply(self, *args, **kwargs):
        if self.name in RANDOM_NUMBER_GEN_OPS:
            return False

        if self.name in SKIP_LIST_OPS:
            return False

        if self.name.startswith("torch.empty"):
            return False

        if is_opname_match(self.name, os.getenv("OP_AUTOCOMPARE_DISABLE_LIST", "")):
            return False

        return is_opname_match(self.name, os.getenv("OP_AUTOCOMPARE_LIST", ".*"))


def dump_all_autocompare_info():
    if len(global_autocompare_result) == 0:
        return
    all_compare_info_list = []
    while len(global_autocompare_result) > 0:
        compare_info = global_autocompare_result.pop(0)
        while len(compare_info["result_list"]) > 0:
            compare_result = compare_info["result_list"].pop(0)
            all_compare_info_list.append({"forward_id": compare_info["forward_id"], **compare_result})

    table = dict_data_list_to_table(all_compare_info_list)
    print(table)
    data_string = table.get_csv_string()
    file_name = f"op_tools_results/op_autocompare_result/op_autocompare_info_pid{os.getpid()}_{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}.csv"  # noqa: E501
    dir = file_name[0 : file_name.rfind("/")]
    os.makedirs(dir, exist_ok=True)

    with open(file_name, "w") as f:
        f.write(data_string)
        f.close
    print(f"op elasped info saved to {file_name}")


atexit.register(dump_all_autocompare_info)
