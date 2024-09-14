# Copyright (c) 2024, DeepLink.
import torch
import re
import importlib
import math
import os
from .pretty_print import dict_data_list_to_table


def traverse_container(container):
    if isinstance(container, dict):
        for key, value in container.items():
            yield from traverse_container(value)
    elif isinstance(container, (list, tuple, set)):
        for item in container:
            yield from traverse_container(item)
    elif type(container).__module__.startswith("torch.return_types"):
        for i in range(len(container)):
            yield container[i]
    else:
        yield container


def is_cpu_op(*args, **kwargs):
    for obj in traverse_container(args):
        if isinstance(obj, torch.Tensor):
            if not obj.device.type == "cpu":
                return False, obj.device

    return True, "cpu"


def transform_contrainer(obj, func):
    if isinstance(obj, (tuple, list, set)):
        return type(obj)([transform_contrainer(v, func) for v in obj])
    elif isinstance(obj, dict):
        return {transform_contrainer(k, func): transform_contrainer(v, func) for k, v in obj.items()}
    elif type(obj).__module__.startswith("torch.return_types"):
        return [transform_contrainer(v) for v in obj]
    else:
        return func(obj)


def to_device(device, obj, detach=False, dtype_cast_dict=dict()):
    def func(obj):
        if isinstance(obj, torch.Tensor):
            if obj.dtype in list(dtype_cast_dict.keys()):
                obj = obj.to(dtype_cast_dict[obj.dtype])
            if detach:
                new_obj = obj.detach().to(device)
                new_obj.requires_grad = obj.requires_grad
            else:
                new_obj = obj.to(device)
            return new_obj
        else:
            return obj

    return transform_contrainer(obj, func)


def is_opname_match(name, op_pattern=None):
    """Determine whether the operator matches the template. The template can be a list of operator names or a regular expression."""
    if name is None:
        return False
    if op_pattern is None:
        return True
    op_list = op_pattern.split(",")
    if name in op_list:
        return True

    for pattern in op_list:
        if name in re.findall(pattern, name):
            return True
    return False


def is_inplace_op(name):
    INPLACES_OP = ["torch.Tensor.__setitem__"]
    return name in INPLACES_OP or (name.endswith("_") and (not name.endswith("__")) and (name.startswith("torch.Tensor.")))


def get_function_from_string(func_str):
    parts = func_str.split(".")
    attrs = [importlib.import_module(parts[0])]
    for i in range(0, len(parts) - 1):
        attr = getattr(attrs[i], parts[i + 1])
        attrs.append(attr)

    return attrs[len(parts) - 1]


def get_dtype_cast_dict_form_str(config):
    """
    'torch.float16->torch.float32,torch.bfloat16->torch.float32' -> {torch.float16:torch.float32, torch.bfloat16:torch.float32}
    """
    dtype_cast_dict = dict()
    if config is not None:
        for item in config.split(","):
            dtype_cast_dict[get_function_from_string(item.split("->")[0])] = get_function_from_string(item.split("->")[1])
    return dtype_cast_dict


VIEW_OPS = [
    "torch.Tensor.reshape",
    "torch.Tensor.adjoint",
    "torch.Tensor.as_strided",
    "torch.Tensor.detach",
    "torch.Tensor.diagonal",
    "torch.Tensor.expand",
    "torch.Tensor.expand_as",
    "torch.Tensor.movedim",
    "torch.Tensor.narrow",
    "torch.Tensor.permute",
    "torch.Tensor.select",
    "torch.Tensor.squeeze",
    "torch.Tensor.transpose",
    "torch.Tensor.t",
    "torch.Tensor.T",
    "torch.Tensor.H",
    "torch.Tensor.mT",
    "torch.Tensor.mH",
    "torch.Tensor.real",
    "torch.Tensor.imag",
    "torch.Tensor.view_as_real",
    "torch.Tensor.unflatten",
    "torch.Tensor.unfold",
    "torch.Tensor.unsqueeze",
    "torch.Tensor.view",
    "torch.Tensor.view_as",
    "torch.Tensor.unbind",
    "torch.Tensor.split",
    "torch.Tensor.hsplit",
    "torch.Tensor.vsplit",
    "torch.Tensor.tensor_split",
    "torch.Tensor.split_with_sizes",
    "torch.Tensor.swapaxes",
    "torch.Tensor.swapdims",
    "torch.Tensor.chunk",
    "torch.Tensor.indices",
    "torch.Tensor.values",
]


def is_view_op(name):
    return name in VIEW_OPS


def tensor_max_diff(a, b):
    a_cpu, b_cpu = a.cpu(), b.cpu()
    if a_cpu.dtype == torch.bool:
        a_cpu = a_cpu.int()
    if b_cpu.dtype == torch.bool:
        b_cpu = b_cpu.int()
    diff = torch.abs(a_cpu - b_cpu)
    max_abs_diff = diff.max().item()
    max_relative_diff = (diff / (a_cpu.abs() + 1e-6)).max().item()
    return max_abs_diff, max_relative_diff


def tensor_allclose(a, b, atol=1e-3, rtol=1e-3):
    a_cpu, b_cpu = a.cpu(), b.cpu()
    try:
        return torch.allclose(a_cpu, b_cpu, atol=atol, rtol=rtol, equal_nan=True)
    except Exception as e:  # noqa: F841
        return False
    return False


def get_error_tolerance(dtype, op_name):
    def get_error_tolerance_for_type(dtype_name, atol, rtol):
        # env priority:
        # OP_NAME_AUTOCOMPARE_ERROR_TOLERANCE_BFLOAT16 > AUTOCOMPARE_ERROR_TOLERANCE_BFLOAT16 > AUTOCOMPARE_ERROR_TOLERANCE
        op_name_processed = op_name.split(".")[-1].upper() + "_"
        env_name = "AUTOCOMPARE_ERROR_TOLERANCE_" + dtype_name.upper()
        high_priority_env_name = op_name_processed + env_name
        if os.getenv(high_priority_env_name) is not None:
            atol, rtol = map(float, os.getenv(high_priority_env_name).split(","))
        elif os.getenv(env_name) is not None:
            atol, rtol = map(float, os.getenv(env_name).split(","))
        elif os.getenv("AUTOCOMPARE_ERROR_TOLERANCE") is not None:
            atol, rtol = map(float, os.getenv("AUTOCOMPARE_ERROR_TOLERANCE").split(","))
        return atol, rtol

    if dtype == torch.float16:
        return get_error_tolerance_for_type("FLOAT16", 1e-4, 1e-4)
    elif dtype == torch.bfloat16:
        return get_error_tolerance_for_type("BFLOAT16", 1e-3, 1e-3)
    elif dtype == torch.float32:
        return get_error_tolerance_for_type("FLOAT32", 1e-5, 1e-5)
    elif dtype == torch.float64:
        return get_error_tolerance_for_type("FLOAT64", 1e-8, 1e-8)
    else:
        atol, rtol = 1e-3, 1e-3
        if os.getenv("AUTOCOMPARE_ERROR_TOLERANCE") is not None:
            atol, rtol = map(float, os.getenv("AUTOCOMPARE_ERROR_TOLERANCE").split(","))
        return atol, rtol


def compare_result(name, a, b):  # noqa: C901
    a_list = []
    b_list = []
    allclose, max_abs_diff, max_relative_diff, error_info = True, 0, 0, ""
    for item in traverse_container(a):
        a_list.append(item)
    for item in traverse_container(b):
        b_list.append(item)

    if len(a_list) != len(b_list):
        error_info += f"Inconsistent output length: {len(a_list)} {len(b_list)}, {a} {b}"
        max_abs_diff = float("nan")
        max_relative_diff = float("nan")
        allclose = False
        return {
            "allclose": allclose,
            "max_abs_diff": max_abs_diff,
            "max_relative_diff": max_relative_diff,
            "error_info": error_info,
            "atol": float("nan"),
            "rtol": float("nan"),
        }
    result_list = []
    for i in range(len(a_list)):
        a_item = a_list[i]
        b_item = b_list[i]
        atol, rtol = 0, 0
        error_info_i = ""
        if a_item is None and b_item is None:
            allclose_i = True
            max_abs_diff_i = 0
            max_relative_diff_i = 0
        elif isinstance(a_item, torch.Tensor) and isinstance(b_item, torch.Tensor):
            if a_item.shape == b_item.shape:
                atol, rtol = get_error_tolerance(a_item.dtype, name)
                max_abs_diff_i, max_relative_diff_i = tensor_max_diff(a_item, b_item)
                allclose_i = tensor_allclose(a_item, b_item, atol=atol, rtol=rtol)
            else:
                error_info_i = f"Inconsistent shape: {a_item.shape} {b_item.shape}"
                max_abs_diff_i = float("nan")
                max_relative_diff_i = float("nan")
                allclose_i = False
            if a_item.dtype != b_item.dtype:
                error_info_i += f"Inconsistent dtypes: {a_item.dtype} {b_item.dtype}"

        elif type(a) != type(b):  # noqa: E721
            error_info_i = f"Inconsistent types: {type(a)} {type(b)}"
            max_abs_diff_i = float("nan")
            max_relative_diff_i = float("nan")
            allclose_i = False
        elif isinstance(a_item, bool):
            allclose_i = a_item == b_item
            max_abs_diff_i = 0.0 if allclose_i else float("nan")
            max_relative_diff_i = 0.0 if allclose_i else float("nan")
            if not allclose_i:
                error_info_i = f" value: {a_item} {b_item} "
        elif isinstance(a_item, (float, int)):
            atol = 1e-6
            rtol = 1e-6
            allclose_i = (math.isnan(a_item) and math.isnan(b_item)) or (abs(a_item - b_item) <= atol + rtol * abs(a_item))
            max_abs_diff_i = abs(a_item - b_item)
            max_relative_diff_i = max_abs_diff_i / (abs(a_item) + 1e-6)
            if not allclose_i:
                error_info_i = f" value: {a_item} {b_item} "
        if len(a_list) > 1:
            prefex = f" {i}th "
        else:
            prefex = ""

        error_info += error_info_i
        allclose = allclose_i and allclose
        max_abs_diff = max(max_abs_diff_i, max_abs_diff)
        max_relative_diff = max(max_relative_diff_i, max_relative_diff)
        result_list.append(
            {
                "name": f"{name + prefex:<30}",
                "allclose": allclose_i,
                "max_abs_diff": f"{max_abs_diff_i:10.9f}",
                "max_relative_diff": f"{max_relative_diff_i:10.9f}",
                "atol": f"{atol:10.9f}",
                "rtol": f"{rtol:10.9f}",
                "error_info": error_info_i,
            }
        )
    print(dict_data_list_to_table(result_list))

    return {
        "allclose": allclose,
        "max_abs_diff": max_abs_diff,
        "max_relative_diff": max_relative_diff,
        "error_info": error_info,
        "atol": atol,
        "rtol": rtol,
        "name": name,
        "result_list": result_list,
    }
