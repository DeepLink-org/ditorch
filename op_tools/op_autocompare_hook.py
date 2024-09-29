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

from .pretty_print import dict_data_list_to_table, packect_data_to_dict_list


SKIP_LIST_OPS = []


class AutoCompareResultCache:
    global_autocompare_result = []

    def __init__(self) -> None:
        self.file_name = f"op_tools_results/op_autocompare_result/op_autocompare_info_pid{os.getpid()}_{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}.csv"  # noqa: E501
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
            self.compare_hook.compare_backward_relate()
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
                self.args_cpu = tuple(args_cpu)
                self.result_cpu = self.func(*self.args_cpu, **self.kwargs_cpu)
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
        self.op_backward_args_to_table(grad_inputs, grad_output)
        self.grad_outputs_cpu = to_device("cpu", grad_output, dtype_cast_dict=self.dtype_cast_dict, detach=True)
        self.grad_inputs_cpu = to_device("cpu", grad_inputs, dtype_cast_dict=self.dtype_cast_dict, detach=True)
        for arg_cpu in traverse_container(self.args_cpu):
            if isinstance(arg_cpu, torch.Tensor) and arg_cpu.grad is not None:
                arg_cpu.grad.zero_()

        self.args_cpu_grad = [None for i in range(self.count_params_with_requires_grad())]

        def post_hook(grad_inputs, grad_outputs):
            self.args_cpu_grad = tuple([grad_input for grad_input in grad_inputs])

        index = -1
        for result_cpu in traverse_container(self.result_cpu):
            index += 1
            if isinstance(result_cpu, torch.Tensor) and result_cpu.requires_grad:
                if result_cpu.grad_fn is None:
                    result_cpu.backward(*self.grad_outputs_cpu)
                    self.args_cpu_grad[index] = result_cpu.grad
                else:
                    handle = result_cpu.grad_fn.register_hook(post_hook)
                    result_cpu.backward(*self.grad_outputs_cpu)
                    handle.remove()

    def register_backward_hook_for_grads(self):
        if self.count_params_with_requires_grad() <= 0:
            self.backward_hook_handle = None
            return
        self.backward_hook_handle = BackwardHookHandle(self)
        for result in traverse_container(self.result):
            if isinstance(result, torch.Tensor):
                if result.grad_fn is not None:
                    self.backward_hook_handle.register_grad_fn_hook(result)

    def compare_forward_result(self):
        compare_info = compare_result(self.name + " output", self.result_device, self.result_cpu)
        compare_result_cache.append(self.forward_op_id, compare_info)

        self.forward_allclose = compare_info["allclose"]
        compare_info["forward_id"] = self.forward_op_id
        return compare_info

    def compare_inputs(self):
        compare_info = compare_result(self.name + " input", self.args, self.args_cpu)
        compare_result_cache.append(self.forward_op_id, compare_info)
        compare_info["forward_id"] = self.forward_op_id
        self.input_allclose = compare_info["allclose"]
        return compare_info

    def compare_forward_relate(self):
        input_compare_result = self.compare_inputs()
        output_compare_result = self.compare_forward_result()

        result_list = input_compare_result["result_list"] + output_compare_result["result_list"]
        print("\n" * 2)
        print(f"{self.name} forward_id: {self.forward_op_id} {self.dtype_cast_dict if len(self.dtype_cast_dict) > 0 else ''}")
        print(self.op_forward_args_to_table())
        print(dict_data_list_to_table(result_list))
        print("\n" * 2)

        self.forward_allclose = self.forward_allclose and self.input_allclose
        if not self.forward_allclose:
            self.save_forward_args()

    def op_forward_args_to_table(self):
        inputs_list = packect_data_to_dict_list(self.name + " inputs", serialize_args_to_dict(*self.args, **self.kwargs))
        output_list = packect_data_to_dict_list(self.name + " outputs", serialize_args_to_dict(self.result))
        cpu_output_list = packect_data_to_dict_list(self.name + " cpu_outputs", serialize_args_to_dict(self.result_cpu))
        forward_args_table = dict_data_list_to_table(inputs_list + output_list + cpu_output_list)
        return forward_args_table

    def op_backward_args_to_table(self, grad_inputs, grad_output):
        grad_inputs_list = packect_data_to_dict_list(self.name + " grad_inputs", serialize_args_to_dict(grad_inputs))
        grad_output_list = packect_data_to_dict_list(self.name + " grad_output", serialize_args_to_dict(grad_output))
        self.backward_args_table = dict_data_list_to_table(grad_output_list + grad_inputs_list)
        return self.backward_args_table

    def count_params_with_requires_grad(self):
        count = 0
        for arg in traverse_container(self.args):
            if isinstance(arg, torch.Tensor) and arg.requires_grad:
                count = count + 1
        return count

    def compare_input_grad(self):
        self.args_grad = self.grad_inputs_cpu
        compare_info = compare_result(self.name + " grad", self.args_cpu_grad, self.args_grad)
        compare_info["forward_id"] = self.forward_op_id
        print(dict_data_list_to_table(compare_info["result_list"]))

        compare_result_cache.append(self.forward_op_id, compare_info)

        self.backward_allclose = compare_info["allclose"]

        return compare_info

    def compare_backward_relate(self):
        backward_compare_result = self.compare_input_grad()

        print("\n" * 2)
        print(f"{self.name} forward_id: {self.forward_op_id}")
        print(self.backward_args_table)
        print(dict_data_list_to_table(backward_compare_result["result_list"]))
        print("\n" * 2)

        if not self.backward_allclose:
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
            self.compare_forward_relate()

            if self.result is None and self.result_cpu is None:
                print(f"{self.name} output is None, no check for backward accuracy")
                return

            self.register_backward_hook_for_grads()
            result = self.result
            id = self.id
            # for reduce device memory usage
            if self.backward_hook_handle is not None:
                self.args = to_device("cpu", self.args, detach=True)
                self.kwargs = to_device("cpu", self.kwargs or {}, detach=True)
                self.result = to_device("cpu", self.result, detach=True)
            else:
                self = None

            gc_cycle = int(os.getenv("OP_AUTOCOMPARE_GARBAGE_COLLECTION_CYCLE", "100"))
            if id % gc_cycle == 0:
                gc.collect()
            return result

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
