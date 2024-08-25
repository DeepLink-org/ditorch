import torch
import types
import os
import filecmp
import json
from collections import OrderedDict
from prettytable import PrettyTable
from collections import namedtuple


class Data:
    @staticmethod
    def load(path):
        return torch.load(path)
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


class CompLayerAcc:
    """compare accuracy
    Args:
        model (nn.Module): model to be compared.
        fixed_layer_inputs (bool): whether to fix the input of the layer or not.
        comp_grad (bool): compare gradient or not.
        dump_or_compare (bool): dump or compare, "dump" means dump the data to file, "compare" means compare the data from file.
        load_data_path (str): the path of the data to be loaded for comparing.
    """
    def __init__(self, model, is_dump_benchmark: bool, load_data_path: str):
        self.model = model
        self.is_dump_benchmark = is_dump_benchmark
        self.load_data_path = load_data_path
        self.saved_datas = set()
        self.sub_dir = "expected" if self.is_dump_benchmark else "real"
        self.data_root_path = os.path.join("accuracy_data",self.sub_dir)
        if not os.path.exists(self.data_root_path):
            os.makedirs(self.data_root_path)
        if not os.path.exists(os.path.join(self.data_root_path, "forward")):
            os.makedirs(os.path.join(self.data_root_path, "forward"))
            os.makedirs(os.path.join(self.data_root_path, "backward"))
        self.layer_names_for_forward = set()
        self.layer_names_for_backward = set()
        self.inspectModules = None
        self.module_full_names = None

    def walk_model(self):
        self.inspectModules = list(walk_modules(self.model))
        [full_name, leaf, module] = self.inspectModules[0]

        for m in self.inspectModules:
            [full_name, leaf, module] = m
            module_full_name = ".".join(full_name)
            # print("======module_full_name:", module_full_name)
            module.register_forward_hook(self._hook_for_dump_args_forward(module_full_name))
            module.register_backward_hook(self._hook_for_dump_args_backward(module_full_name))

    def insert_hook(self):
        print(self.model)
        self.walk_model()
        self.module_full_names = [".".join(m.full_name) for m in self.inspectModules]
        save_list_as_txt(self.module_full_names, os.path.join(self.data_root_path, "module_full_names.txt"))
        return self.model

    def _hook_for_dump_args_forward(self, layer_name):
        def true_hook(module, args, output):
            data = {"layer_name":layer_name, "inputs": args, "output": output}
            Data.save(data, os.path.join(self.data_root_path, "forward", layer_name + ".pt"))
        return true_hook

    def _hook_for_dump_args_backward(self, layer_name):
        def true_hook(module, grad_input, grad_output):
            data = {"layer_name":layer_name, "grad_input": grad_input, "grad_output": grad_output}
            Data.save(data, os.path.join(self.data_root_path, "backward", layer_name + ".pt"))
        return true_hook

def compare(data_expected, data_real, cmp_res_list):
    # all data saved in cmp_res_list, a trick for pass ref to function
    assert len(cmp_res_list) == 1, "cmp_res_list should only have one element"
    assert isinstance(cmp_res_list[0], OrderedDict), "cmp_res_list[0] should be a OrderedDict"
    cmp_res = cmp_res_list[0]
    atol = 1e-3
    rtol = 1e-3
    if data_expected is None or data_real is None:
        assert data_expected == data_real, f"the data_expected {data_expected} and data_real {data_real} is not equal, and one of them is None"
        cmp_res["allclose"] = 1
        return

    if isinstance(data_expected, torch.Tensor) and isinstance(data_real, torch.Tensor):
        assert data_expected.dtype == data_real.dtype, f"the dtype of data_expected {data_expected.dtype} and data_real {data_real.dtype} is not equal."
        assert data_expected.shape == data_real.shape, f"the shape of data_expected {data_expected.shape} and data_real {data_real.shape} is not equal."

        data_expected = data_expected.detach().cpu().float()
        data_real = data_real.detach().cpu().float()
        is_allclose = torch.allclose(data_expected, data_real, atol, rtol, equal_nan=True)
        cmp_res["allclose"] =  1 if is_allclose else 0
        if not is_allclose:
            diff = data_real - data_expected
            max_diff = torch.max(torch.abs(diff))
            max_index = torch.argmax(diff)
            max_diff_data_expected = data_expected.flatten()[max_index]
            max_diff_data_real = data_real.flatten()[max_index]
            relative_err = (max_diff_data_real - max_diff_data_expected) / max_diff_data_expected
            cmp_res["err"] = f"max_diff: {max_diff}, expected: {max_diff_data_expected}, real: {max_diff_data_real}, relative err: {relative_err:.6f}"

    elif isinstance(data_expected,(list,tuple)) and isinstance(data_real,(list,tuple)):
        assert len(data_expected) == len(
            data_real
        ), "the data_expected and data_real is type of tuple or list, but the length of data_expected and data_real is not equal"
        for i in range(len(data_expected)):
            cmp_res[i] = OrderedDict()
            compare(data_expected[i], data_real[i], [cmp_res[i]])

    elif isinstance(data_expected, str) or isinstance(data_real, str):
        assert type(data_expected) == type(data_real), "the data_expected or data_real is type of str, but the type of data_expected and data_real is not equal"
        is_allclose = data_expected == data_real
        cmp_res["allclose"] = 1 if is_allclose else 0
        if not is_allclose:
            cmp_res["data_expected"] = data_expected
            cmp_res["data_real"] = data_real
    else:
        is_allclose = abs(data_real - data_expected) <= atol + rtol * abs(data_expected)
        cmp_res["allclose"] =  1 if is_allclose else 0
        if not is_allclose:
            relative_err = (data_real - data_expected) / data_expected
            cmp_res["err"] = (
                f"diff: {data_real - data_expected}, expected: {data_expected}, real: {data_real}, relative err: {relative_err:.6f}"
            )


def compare_accuracy(data_expected_dir="accuracy_data/expected", data_real_dir="accuracy_data/real"):
    table = PrettyTable(["No.", "layer_name", "forward", "backward" ])
    table.align = "l"
    cmp_res = {}

    module_name_path_expected = os.path.join(data_expected_dir, "module_full_names.txt")
    module_name_path_real = os.path.join(data_real_dir, "module_full_names.txt")

    assert os.path.exists(module_name_path_expected), " module_full_name.txt is not exist in {data_expected_dir}"
    assert os.path.exists(module_name_path_real), " module_full_name.txt is not exist in {data_real_dir}"

    assert filecmp.cmp(module_name_path_expected, module_name_path_real, shallow=False), "the two module_full_name.txt is not equal"
    full_name_modules = load_list_from_txt(module_name_path_real)
    for i in range(len(full_name_modules)):
        full_name = full_name_modules[i]
        # cmp forward
        expected_forward_path = os.path.join(data_expected_dir, "forward", full_name + ".pt")
        real_forward_path = os.path.join(data_real_dir, "forward", full_name + ".pt")
        assert os.path.exists(real_forward_path) == os.path.exists(
            expected_forward_path
        ), "one of the path {expected_forward_path} and {real_forward_path}, but the other is not exist."

        output_cmp_res = OrderedDict()
        if os.path.exists(expected_forward_path):
            forward_data_expected = Data.load(expected_forward_path)
            forward_data_real = Data.load(real_forward_path)
            compare(forward_data_expected['output'], forward_data_real["output"], [output_cmp_res])
        cmp_res["output"]  = output_cmp_res

        # cmp backward
        expected_backward_path = os.path.join(data_expected_dir, "backward", full_name + ".pt")
        real_backward_path = os.path.join(data_real_dir, "backward", full_name + ".pt")
        assert os.path.exists(real_backward_path) == os.path.exists(
            expected_backward_path
        ), "one of the path {expected_backward_path} and {real_backward_path}, but the other is not exist."

        grad_input_cmp_res = OrderedDict()
        if os.path.exists(expected_backward_path):
            backward_data_expected = Data.load(expected_backward_path)
            backward_data_real = Data.load(real_backward_path)
            compare(backward_data_expected['grad_input'], backward_data_real["grad_input"], [grad_input_cmp_res])
        cmp_res["grad_input"] = grad_input_cmp_res

        # to str
        output_cmp_res_str = json.dumps(output_cmp_res)
        grad_input_cmp_res_str = json.dumps(grad_input_cmp_res)
        table.add_row([i, full_name, output_cmp_res_str, grad_input_cmp_res_str])
    print(table)
