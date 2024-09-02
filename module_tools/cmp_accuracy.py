import torch
import types
import os
import filecmp
import json
from collections import OrderedDict
from prettytable import PrettyTable
from collections import namedtuple
from typing import Union
import fcntl


lock_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dir_lock.lock")


def safe_mkdir_with_lock(path, lock_file=lock_file):
    assert os.path.exists(lock_file), "the lock file does not exist."
    with open(lock_file, "w") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX)  # 获得独占锁
            except BlockingIOError:
                continue
            if not os.path.exists(path):
                os.makedirs(path)
            fcntl.flock(lock, fcntl.LOCK_UN)  # 释放锁
            break


class Data:
    @staticmethod
    def load(path, map_location=None):
        return torch.load(path, map_location=map_location)

    @staticmethod
    def save(data, path):
        torch.save(data, path)


InspectModule = namedtuple("InspectModule", ["full_name", "leaf", "module"])


def walk_modules(module, name="", full_name=()):
    """Generator. Walks through a PyTorch Module and outputs Trace tuples"""
    if not name:
        name = module.__class__.__name__
    named_children = list(module.named_children())
    full_name = full_name + (name,)
    yield InspectModule(full_name, len(named_children) == 0, module)
    # recursively walk into all submodules
    for name, child_module in named_children:
        yield from walk_modules(child_module, name=name, full_name=full_name)


def save_list_as_txt(li_str: list[str], path):
    with open(path, "w") as f:
        for item in li_str:
            f.write(f"{item}\n")


def load_list_from_txt(path):
    with open(path, "r") as f:
        return [line.strip() for line in f]


def for_each_tensor(
    inputs: Union[list[torch.Tensor], tuple[torch.Tensor], torch.Tensor, int, float],
    func,
):
    if isinstance(inputs, torch.Tensor):
        return func(inputs)
    elif isinstance(inputs, (list, tuple)):
        return tuple([for_each_tensor(item, func) for item in inputs])
    else:
        return inputs


def clone_tensors(
    inputs: Union[list[torch.Tensor], tuple[torch.Tensor], torch.Tensor, int, float]
):
    return for_each_tensor(inputs, lambda x: x.clone())


def tensors_to_cpu(
    inputs: Union[list[torch.Tensor], tuple[torch.Tensor], torch.Tensor, int, float]
):
    return for_each_tensor(inputs, lambda x: x.cpu())


def tensors_to_cuda(
    inputs: Union[list[torch.Tensor], tuple[torch.Tensor], torch.Tensor, int, float]
):
    return for_each_tensor(inputs, lambda x: x.cuda())


class CompLayerAcc:
    def __init__(
        self, model: torch.nn.Module, is_dump_benchmark: bool, is_fixed_input: bool
    ):
        """Compare the accuracy of the forward and backward of the model layer by layer.

        Args:
            model (torch.nn.Module): The model to be compared.
            is_dump_benchmark (bool): Whether to dump the benchmark data.
        """
        if is_fixed_input:
            assert not is_dump_benchmark, "please dump the input first."
        self.rank = os.environ["RANK"]
        self.model = model
        self.is_dump_benchmark = is_dump_benchmark
        self.saved_datas = set()
        # self.top_dir = "accuracy_data"
        self.top_dir = "/deeplink_afs/wangxing/accuracy_data"
        self.sub_dir = "expected" if self.is_dump_benchmark else "real"
        self.data_root_path = os.path.join(
            self.top_dir, f"rank{self.rank}", self.sub_dir
        )
        if not os.path.exists(self.data_root_path):
            safe_mkdir_with_lock(self.data_root_path)
        self.forward_path = os.path.join(self.data_root_path, "forward")
        self.backward_path = os.path.join(self.data_root_path, "backward")
        self.is_fixed_input = is_fixed_input
        self.fixed_input_path = os.path.join(
            self.top_dir, f"rank{self.rank}", "expected"
        )
        if self.is_fixed_input:
            assert os.path.exists(
                self.fixed_input_path
            ), "the fixed input path doesn't exist."
        if not os.path.exists(self.forward_path):
            os.makedirs(self.forward_path)
        if not os.path.exists(self.backward_path):
            os.makedirs(self.backward_path)

        self.modules_full_name_txt_path = os.path.join(
            self.data_root_path, f"modules_full_name.txt"
        )
        self.modules_full_name_run_txt_path = os.path.join(
            self.data_root_path, f"modules_full_name_run.txt"
        )
        self.modules_full_name = None
        self.modules_full_name_run = []

        self.layer_names_for_forward = set()
        self.layer_names_for_backward = set()
        self.inspectModules = None
        self.is_1st_backward = True

    def walk_model(self):
        self.inspectModules = list(walk_modules(self.model))
        [full_name, leaf, module] = self.inspectModules[0]

        for m in self.inspectModules:
            [full_name, leaf, module] = m
            module_full_name = ".".join(full_name)
            module.register_forward_hook(
                self._hook_for_dump_args_forward(module_full_name)
            )
            module.register_full_backward_hook(
                self._hook_for_dump_args_backward(module_full_name)
            )
            if self.is_fixed_input:
                module.register_forward_pre_hook(
                    self._hook_for_change_input_forward(module_full_name)
                )
                # if module_full_name != "PipelineEngine.module":  # Comment out code as you see fit
                #     module.register_full_backward_pre_hook(
                #         self._hook_for_change_input_backward(module_full_name)
                # )

    def insert_hook(self):
        print(self.model)
        self.walk_model()
        self.modules_full_name = [".".join(m.full_name) for m in self.inspectModules]
        save_list_as_txt(self.modules_full_name, self.modules_full_name_txt_path)
        return self.model

    def _hook_for_change_input_forward(self, layer_name):
        def true_hook(module, input):
            input_data = Data.load(
                os.path.join(self.fixed_input_path, "forward", f"{layer_name}.pt")
            )
            return tensors_to_cuda(input_data["input"])

        return true_hook

    def _hook_for_change_input_backward(self, layer_name):
        def true_hook(module, grad_output):
            input_data = Data.load(
                os.path.join(self.fixed_input_path, "backward", f"{layer_name}.pt")
            )
            return tensors_to_cuda(input_data["grad_output"])

        return true_hook

    def _hook_for_dump_args_forward(self, layer_name):
        def true_hook(module, input, output):
            self.modules_full_name_run.append(layer_name)
            data = {"layer_name": layer_name, "input": input, "output": output}
            save_path = os.path.join(self.forward_path, f"{layer_name}.pt")
            Data.save(tensors_to_cpu(data), save_path)

        return true_hook

    def _hook_for_dump_args_backward(self, layer_name):
        def true_hook(module, grad_input, grad_output):
            if self.is_1st_backward:
                save_list_as_txt(
                    self.modules_full_name_run, self.modules_full_name_run_txt_path
                )
                self.is_1st_backward = False
            data = {
                "layer_name": layer_name,
                "grad_input": grad_input,
                "grad_output": grad_output,
            }
            save_path = os.path.join(self.backward_path, f"{layer_name}.pt")
            Data.save(tensors_to_cpu(data), save_path)

        return true_hook


def compare(data_expected, data_real, cmp_res_list, rtol, atol):
    # all data saved in cmp_res_list, a trick for pass ref to function
    assert len(cmp_res_list) == 1, "cmp_res_list should only have one element"
    assert isinstance(
        cmp_res_list[0], OrderedDict
    ), "cmp_res_list[0] should be a OrderedDict"
    cmp_res = cmp_res_list[0]
    if data_expected is None or data_real is None:
        assert (
            data_expected == data_real
        ), f"the data_expected {data_expected} and data_real {data_real} is not equal, and one of them is None"
        cmp_res["allclose"] = 1
        return

    if isinstance(data_expected, torch.Tensor) and isinstance(data_real, torch.Tensor):
        assert (
            data_expected.dtype == data_real.dtype
        ), f"the dtype of data_expected {data_expected.dtype} and data_real {data_real.dtype} is not equal."
        assert (
            data_expected.shape == data_real.shape
        ), f"the shape of data_expected {data_expected.shape} and data_real {data_real.shape} is not equal."

        data_expected = data_expected.detach().cpu().float()
        data_real = data_real.detach().cpu().float()
        is_allclose = torch.allclose(
            data_expected, data_real, atol, rtol, equal_nan=True
        )
        cmp_res["allclose"] = 1 if is_allclose else 0
        if not is_allclose:
            diff_abs = torch.abs(data_real - data_expected)
            max_index = torch.argmax(diff_abs)
            max_diff = torch.max(diff_abs)
            max_diff_data_expected = data_expected.flatten()[max_index]
            max_diff_data_real = data_real.flatten()[max_index]
            relative_err = (
                max_diff_data_real - max_diff_data_expected
            ) / max_diff_data_expected
            cmp_res["err"] = (
                f"max_diff: {max_diff:.6f}, expected: {max_diff_data_expected:.6f}, real: {max_diff_data_real:.6f}, relative err: {relative_err:.6f}"
            )

    elif isinstance(data_expected, (list, tuple)) and isinstance(
        data_real, (list, tuple)
    ):
        assert len(data_expected) == len(
            data_real
        ), "the data_expected and data_real is type of tuple or list, but the length of data_expected and data_real is not equal"
        for i in range(len(data_expected)):
            cmp_res[i] = OrderedDict()
            compare(data_expected[i], data_real[i], [cmp_res[i]], rtol, atol)

    elif isinstance(data_expected, str) or isinstance(data_real, str):
        assert type(data_expected) == type(
            data_real
        ), "the data_expected or data_real is type of str, but the type of data_expected and data_real is not equal"
        is_allclose = data_expected == data_real
        cmp_res["allclose"] = 1 if is_allclose else 0
        if not is_allclose:
            cmp_res["data_expected"] = data_expected
            cmp_res["data_real"] = data_real
    else:
        is_allclose = abs(data_real - data_expected) <= atol + rtol * abs(data_expected)
        cmp_res["allclose"] = 1 if is_allclose else 0
        if not is_allclose:
            relative_err = (data_real - data_expected) / data_expected
            cmp_res["err"] = (
                f"diff: {data_real - data_expected:.6f}, expected: {data_expected:.6f}, real: {data_real:.6f}, relative err: {relative_err:.6f}"
            )


def single_process_cmp_accuracy(
    data_expected_dir="accuracy_data/expected",
    data_real_dir="accuracy_data/real",
    rank=0,
):
    rtol = 1e-3
    atol = 1e-3
    table = PrettyTable(
        [
            f"No.(rank{rank})",
            "layer_name",
            f"forward: output(atol:{atol}, rtol:{rtol})",
            f"backward: grad_input(atol:{atol}, rtol:{rtol})",
        ]
    )
    table.align = "l"
    cmp_res = {}

    module_name_path_expected = os.path.join(
        data_expected_dir, "modules_full_name_run.txt"
    )
    module_name_path_real = os.path.join(data_real_dir, "modules_full_name_run.txt")

    assert os.path.exists(
        module_name_path_expected
    ), " modules_full_name_run.txt is not exist in {data_expected_dir}"
    assert os.path.exists(
        module_name_path_real
    ), " modules_full_name_run.txt is not exist in {data_real_dir}"
    modules_expected = load_list_from_txt(module_name_path_expected)
    modules_real = load_list_from_txt(module_name_path_real)
    modules_intersection = list(filter(lambda x: x in modules_real, modules_expected))
    modules_only_in_expected = list(
        filter(lambda x: x not in modules_intersection, modules_expected)
    )
    modules_only_in_real = list(
        filter(lambda x: x not in modules_intersection, modules_real)
    )
    for i in range(len(modules_intersection)):
        full_name = modules_intersection[i]
        # cmp forward
        expected_forward_path = os.path.join(
            data_expected_dir, "forward", full_name + ".pt"
        )
        real_forward_path = os.path.join(data_real_dir, "forward", full_name + ".pt")
        assert os.path.exists(real_forward_path) == os.path.exists(
            expected_forward_path
        ), "one of the path {expected_forward_path} and {real_forward_path}, but the other is not exist."

        output_cmp_res = OrderedDict()
        if os.path.exists(expected_forward_path):
            forward_data_expected = Data.load(expected_forward_path, map_location="cpu")
            forward_data_real = Data.load(real_forward_path, map_location ="cpu")
            compare(
                forward_data_expected["output"],
                forward_data_real["output"],
                [output_cmp_res],
                rtol,
                atol,
            )
        cmp_res["output"] = output_cmp_res

        # cmp backward
        expected_backward_path = os.path.join(
            data_expected_dir, "backward", full_name + ".pt"
        )
        real_backward_path = os.path.join(data_real_dir, "backward", full_name + ".pt")
        if os.path.exists(real_backward_path):
            grad_input_cmp_res = OrderedDict()
            if os.path.exists(expected_backward_path):
                backward_data_expected = Data.load(expected_backward_path, map_location="cpu")
                backward_data_real = Data.load(real_backward_path, map_location="cpu")
                compare(
                    backward_data_expected["grad_input"],
                    backward_data_real["grad_input"],
                    [grad_input_cmp_res],
                    rtol,
                    atol,
                )
            cmp_res["grad_input"] = grad_input_cmp_res
        else:
            grad_input_cmp_res = {}
        # to str
        output_cmp_res_str = json.dumps(output_cmp_res)
        grad_input_cmp_res_str = json.dumps(grad_input_cmp_res)
        table.add_row([i, full_name, output_cmp_res_str, grad_input_cmp_res_str])
    print(table)
    if modules_only_in_expected:
        print("Modules that exist only in the expected module:")
        print("\n".join([" " * 4 + m_str for m_str in modules_only_in_expected]))
    if modules_only_in_real:
        print("Modules that exist only in the real module:"),
        print("\n".join([" " * 4 + m_str for m_str in modules_only_in_real]))
    print("=" * 80)


def compare_accuracy(data_dir="accuracy_data"):
    sub_dirs = [
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    ]
    for dir in sub_dirs:
        rank = int(dir[4:])
        sub_dir = os.path.join(data_dir, dir)
        single_process_cmp_accuracy(
            os.path.join(sub_dir, "expected"), os.path.join(sub_dir, "real"), rank
        )
