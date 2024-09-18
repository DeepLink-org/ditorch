# Copyright (c) 2024, DeepLink.
import os
import torch
import time
import atexit
from .base_hook import BaseHook, DisableHookGuard

from .save_op_args import serialize_args_to_dict
from .utils import is_opname_match
from .pretty_print import (
    pretty_print_op_args,
    dict_data_list_to_table,
    packect_data_to_dict_list,
)

global_elasped_info_dict = dict()


class TimeMeasureResultCache:
    global_elasped_info_dict = dict()

    def __init__(self) -> None:
        self.file_name = f"op_tools_results/op_time_measure_result/op_elasped_info_pid{os.getpid()}_{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}.csv"  # noqa: E501
        self.dir = self.file_name[0 : self.file_name.rfind("/")]
        self.ordered_keys = [
            "name",
            "forward_id",
            "forward_elasped",
            "backward_elasped",
            "unit",
            "input",
            "output",
            "grad_inputs",
            "grad_outputs",
        ]

    def append(self, forward_id, elasped_info):
        if forward_id not in self.global_elasped_info_dict:
            self.global_elasped_info_dict[forward_id] = elasped_info
        else:
            self.global_elasped_info_dict[forward_id].update(elasped_info)

        if len(self.global_elasped_info_dict) > 1000:
            self.write_to_file()

    def write_to_file(self):
        if len(self.global_elasped_info_dict) == 0:
            return
        simple_data_list = []
        for key, value in self.global_elasped_info_dict.items():
            new_value = {k: value[k] for k in self.ordered_keys}
            simple_value = {k: value[k] for k in self.ordered_keys[0:5]}
            simple_data_list.append(simple_value)
            self.global_elasped_info_dict[key] = new_value

        print(dict_data_list_to_table(simple_data_list))

        table = dict_data_list_to_table(list(self.global_elasped_info_dict.values()))
        self.global_elasped_info_dict.clear()
        data_string = table.get_csv_string()
        os.makedirs(self.dir, exist_ok=True)

        with open(self.file_name, "a+") as f:
            f.write(data_string)
            f.close
        print(f"op elasped info saved to {self.file_name}")


time_measure_result_cache = TimeMeasureResultCache()


def dump_all_op_elasped_info():
    time_measure_result_cache.write_to_file()


class BackwardHookHandle:
    def __init__(self, name, id) -> None:
        self.name = name
        self.id = id

    def grad_fun_prehook(self):
        def grad_fun(grad_inputs):
            torch.cuda.current_stream().synchronize()
            self.start_time = time.time()

        return grad_fun

    def grad_fun_posthook(self):
        def grad_fun(grad_inputs, grad_outputs):
            torch.cuda.current_stream().synchronize()
            self.end_time = time.time()
            self.backward_elasped = self.end_time - self.start_time
            data_dict_list = []
            data_dict_list += packect_data_to_dict_list(self.name, serialize_args_to_dict(grad_outputs), prefix="grad_outputs ")
            data_dict_list += packect_data_to_dict_list(self.name, serialize_args_to_dict(grad_inputs), prefix="grad_inputs  ")
            table = dict_data_list_to_table(data_dict_list)
            print(table)
            elasped_info_dict = {
                "name": self.name,
                "forward_id": self.id,
                "backward_elasped": f"{(self.backward_elasped * 1000):>10.8f}",
                "unit": "ms",
            }
            print(dict_data_list_to_table([elasped_info_dict]))
            elasped_info_dict["grad_inputs"] = serialize_args_to_dict(grad_inputs)
            elasped_info_dict["grad_outputs"] = serialize_args_to_dict(grad_outputs)
            time_measure_result_cache.append(self.id, elasped_info_dict)

        return grad_fun


class OpTimeMeasureHook(BaseHook):
    def __init__(self, name, func) -> None:
        super().__init__(name, func)

    def before_call_op(self, *args, **kwargs):
        torch.cuda.current_stream().synchronize()
        self.start_time = time.time()

    def after_call_op(self, result):
        torch.cuda.current_stream().synchronize()
        self.end_time = time.time()
        self.foward_elasped = self.end_time - self.start_time

        self.backward_hook_handle = BackwardHookHandle(self.name, self.id)
        if isinstance(self.result, torch.Tensor):
            if self.result.grad_fn is not None:
                self.result.grad_fn.register_hook(self.backward_hook_handle.grad_fun_posthook())
                self.result.grad_fn.register_prehook(self.backward_hook_handle.grad_fun_prehook())
        elif isinstance(self.result, (tuple, list)) or type(self.result).__module__.startswith("torch.return_types"):
            # torch.return_types is a structseq, aka a "namedtuple"-like thing defined by the Python C-API.
            for i in range(len(self.result)):
                if isinstance(self.result[i], torch.Tensor) and self.result[i].grad_fn is not None:
                    self.result[i].grad_fn.register_hook(self.backward_hook_handle.grad_fun_posthook())

                    self.result[i].grad_fn.register_prehook(self.backward_hook_handle.grad_fun_prehook())

        with DisableHookGuard():
            pretty_print_op_args(
                self.name,
                serialize_args_to_dict(*self.args, **self.kwargs),
                serialize_args_to_dict(self.result),
            )
            elasped_info_dict = {
                "name": self.name,
                "forward_id": self.id,
                "forward_elasped": f"{(self.foward_elasped * 1000):>10.8f}",
                "unit": "ms",
            }
            print(dict_data_list_to_table([elasped_info_dict]))
            elasped_info_dict["input"] = serialize_args_to_dict(*self.args, **self.kwargs)
            elasped_info_dict["output"] = serialize_args_to_dict(self.result)
            time_measure_result_cache.append(self.id, elasped_info_dict)

    def is_should_apply(self, *args, **kwargs):
        if is_opname_match(self.name, os.getenv("OP_TIME_MEASURE_DISABLE_LIST", "")):
            return False

        return is_opname_match(self.name, os.getenv("OP_TIME_MEASURE_LIST", ".*"))


atexit.register(dump_all_op_elasped_info)
