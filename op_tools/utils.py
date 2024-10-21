# Copyright (c) 2024, DeepLink.
import torch
import re
import importlib
import math
import os
import gc
import traceback
import psutil

import tracemalloc

tracemalloc.start()


def traverse_container(container):
    if isinstance(container, dict):
        for key, value in container.items():
            yield from traverse_container(key)
            yield from traverse_container(value)
    elif isinstance(container, (list, tuple, set)):
        for item in container:
            yield from traverse_container(item)
    elif isinstance(container, (int, float, str, bool, str, torch.Tensor)):
        yield container
    else:
        try:
            for i in range(len(container)):
                yield from traverse_container(container[i])
        except Exception as e:  # noqa: F841
            yield container


def is_cpu_op(*args, **kwargs):
    for obj in traverse_container(args):
        if isinstance(obj, torch.Tensor):
            if obj.device.type != "cpu":
                return False, obj.device
        elif isinstance(obj, torch.device):
            if obj.type != "cpu":
                return False, obj
        elif isinstance(obj, str):
            try:
                device = torch.device(obj)
                if device.type != "cpu":
                    return False, device
            except Exception as e:  # noqa: F841
                pass
    if kwargs.get("device", None) is not None:
        device = torch.device(kwargs["device"])
        return device.type == "cpu", device

    return True, torch.device("cpu")


def transform_contrainer(obj, func):
    if isinstance(obj, (tuple, list, set)):
        cls = tuple if isinstance(obj, tuple) else type(obj)  # instance(torch.return_types, tuple) is True
        return cls([transform_contrainer(v, func) for v in obj])
    elif isinstance(obj, dict):
        return {transform_contrainer(k, func): transform_contrainer(v, func) for k, v in obj.items()}
    else:
        return func(obj)


def is_inf_or_nan(x):
    def func(obj):
        if isinstance(obj, torch.Tensor):
            return obj.isinf().any().item() or obj.isnan().any().item()
        elif isinstance(obj, (float, int, bool)):
            return math.isinf(obj) or math.isnan(obj)
        else:
            return False
    return transform_contrainer(x, func)


def compute_tensor_features(arg):
    arg = arg.detach()
    arg_cpu = arg.cpu()
    arg_cpu_float = arg_cpu.float()
    return {
        "inf_or_nan": is_inf_or_nan(arg) or is_inf_or_nan(arg_cpu),
        "min": arg_cpu_float.min().item(),
        "max": arg_cpu_float.max().item(),
        "mean": arg_cpu_float.mean().item(),
        "std": arg_cpu_float.std().item(),
        "norm": arg_cpu_float.norm().item(),
    }


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
        elif isinstance(obj, torch.device):
            return torch.device(device)
        elif isinstance(obj, str):
            try:
                _ = torch.device(obj)
                return str(device)
            except Exception as e:  # noqa: F841
                return obj
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
    INPLACES_OP = ["torch.Tensor.__setitem__", "torch.Tensor.to", "torch.Tensor.contiguous", "torch.Tensor.to"]
    return name in INPLACES_OP or (name.endswith("_") and (not name.endswith("__")) and (name.startswith("torch.Tensor.")))


def get_function_from_string(func_str):
    parts = func_str.split(".")
    attrs = [importlib.import_module(parts[0].strip())]
    for i in range(0, len(parts) - 1):
        attr = getattr(attrs[i], parts[i + 1].strip())
        attrs.append(attr)

    return attrs[len(parts) - 1]


def get_dtype_cast_dict_form_str(config):
    """
    'torch.float16->torch.float32,torch.bfloat16->torch.float32' -> {torch.float16:torch.float32, torch.bfloat16:torch.float32}
    """
    dtype_cast_dict = dict()
    if config is not None:
        for item in config.split(","):
            item = item.strip()
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
    "torch.rand_like",
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


def is_view_op(name):
    return name in VIEW_OPS


def is_random_number_gen_op(name):
    return name in RANDOM_NUMBER_GEN_OPS


def is_dtype_cast_op(name, *args, **kwargs):
    if "dtype" in kwargs.keys() and kwargs["dtype"] is not None:
        return True
    for arg in args:
        if isinstance(arg, torch.dtype):
            return True
    dtype_cast_op = [
        "torch.Tensor.double",
        "torch.Tensor.float",
        "torch.Tensor.half",
        "torch.Tensor.bfloat16",
        "torch.Tensor.long",
        "torch.Tensor.int",
        "torch.Tensor.short",
        "torch.Tensor.bool",
    ]
    if name in dtype_cast_op:
        return True
    return False


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


def tensor_cos_similarity(a, b):
    a_cpu, b_cpu = a.cpu().float(), b.cpu().float()
    cos_sim = torch.nn.functional.cosine_similarity(a_cpu.reshape(-1), b_cpu.reshape(-1), dim=-1)
    return cos_sim.item()


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


def compare_result(name, a, b, ignore_index=None):  # noqa: C901
    a_list = []
    b_list = []
    allclose, max_abs_diff, max_relative_diff, error_info, atol, rtol, cos_similarity = True, 0, 0, "", 0, 0, -1e8
    for item in traverse_container(a):
        a_list.append(item)
    for item in traverse_container(b):
        b_list.append(item)

    result_list = []
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
            "result_list": result_list,
        }
    for i in range(len(a_list)):
        if i == ignore_index:
            continue
        a_item = a_list[i]
        b_item = b_list[i]
        atol_i, rtol_i = 0, 0
        error_info_i = ""
        cos_similarity_i = None
        if a_item is None and b_item is None:
            allclose_i = True
            max_abs_diff_i = 0
            max_relative_diff_i = 0
        elif isinstance(a_item, torch.Tensor) and isinstance(b_item, torch.Tensor):
            if a_item.dtype != b_item.dtype:
                error_info_i += f"Inconsistent dtypes: {a_item.dtype} {b_item.dtype}"
                b_item = b_item.to(a_item.dtype)
            if a_item.shape == b_item.shape:
                atol_i, rtol_i = get_error_tolerance(a_item.dtype, name)
                if a_item.numel() > 0:
                    max_abs_diff_i, max_relative_diff_i = tensor_max_diff(a_item, b_item)
                    allclose_i = tensor_allclose(a_item, b_item, atol=atol_i, rtol=rtol_i)
                    cos_similarity_i = tensor_cos_similarity(a_item, b_item)
                else:
                    max_abs_diff_i, max_relative_diff_i = 0.0, 0.0
                    allclose_i = True
            else:
                error_info_i = f"Inconsistent shape: {a_item.shape} {b_item.shape}"
                max_abs_diff_i = float("nan")
                max_relative_diff_i = float("nan")
                allclose_i = False

        elif type(a_item) != type(b_item):  # noqa: E721
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
            atol_i = 1e-6
            rtol_i = 1e-6
            allclose_i = (math.isnan(a_item) and math.isnan(b_item)) or (abs(a_item - b_item) <= atol_i + rtol_i * abs(b_item))
            max_abs_diff_i = abs(a_item - b_item)
            max_relative_diff_i = max_abs_diff_i / (abs(b_item) + 1e-6)
            if not allclose_i:
                error_info_i = f" value: {a_item} {b_item} "
        else:
            try:
                allclose_i = a_item == b_item
            except Exception as e:
                allclose_i = False
                error_info_i = str(e)
            error_info_i += f" value: {a_item} {b_item}"
            if not allclose_i:
                max_abs_diff_i = float("nan")
                max_relative_diff_i = float("nan")
            else:
                max_abs_diff_i = 0
                max_relative_diff_i = 0
        if len(a_list) > 1:
            prefex = f"[{i}]"
        else:
            prefex = ""

        error_info += error_info_i
        allclose = allclose_i and allclose
        max_abs_diff = max(max_abs_diff_i, max_abs_diff)
        max_relative_diff = max(max_relative_diff_i, max_relative_diff)
        atol = max(atol_i, atol)
        rtol = max(rtol_i, rtol)
        if cos_similarity_i is None:
            cos_similarity_i = 1 if allclose_i else -1
        cos_similarity = max(cos_similarity, cos_similarity_i)
        result_list.append(
            {
                "name": f"{name + prefex:<30}",
                "allclose": allclose_i,
                "cosine_similarity": f"{cos_similarity_i:1.9f}",
                "max_abs_diff": f"{max_abs_diff_i:10.9f}",
                "max_relative_diff": f"{max_relative_diff_i:10.9f}",
                "atol": f"{atol_i:10.9f}",
                "rtol": f"{rtol_i:10.9f}",
                "error_info": error_info_i,
            }
        )

    return {
        "allclose": allclose,
        "cos_similarity": cos_similarity,
        "max_abs_diff": max_abs_diff,
        "max_relative_diff": max_relative_diff,
        "error_info": error_info,
        "atol": atol,
        "rtol": rtol,
        "name": name,
        "result_list": result_list,
    }


class GarbageCollectEvaluate:
    def __init__(self) -> None:
        self.rss = psutil.Process().memory_info().rss
        self.device_memory_usage = torch.cuda.memory_allocated()
        self.current_rss = psutil.Process().memory_info().rss
        self.current_device_memory_usage = torch.cuda.memory_allocated()
        self.max_diff = 20 << 30
        self.id = 0

    def is_shoule_collect(self):
        self.id += 1
        if self.id % 2 == 0:
            return False
        self.current_rss = psutil.Process().memory_info().rss
        self.current_device_memory_usage = torch.cuda.memory_allocated()
        print(f"GarbageCollectEvaluate:  host_memory_usage: {self.current_rss >> 20} MB, device_memory_usage: {self.current_device_memory_usage >> 20} MB, device_memory_reserved: {torch.cuda.memory_reserved() >> 20} MB")  # noqa: E501
        if (self.current_rss - self.rss > 2 * self.max_diff) or \
           (self.id % 100 == 0) or \
           (self.current_device_memory_usage - self.device_memory_usage > self.max_diff):
            self.id = 0
            return True
        else:
            return False

    def collect(self):
        gc.collect()
        torch.cuda.empty_cache()
        self.rss = psutil.Process().memory_info().rss
        self.device_memory_usage = torch.cuda.memory_allocated()
        print(
            f"GarbageCollectEvaluate: after collect : rss: {self.rss >> 20} MB, current_rss: {self.current_rss >> 20} MB, max_diff: {self.max_diff>>20} MB, device_memory_usage: {self.device_memory_usage >> 20} MB, current_device_memory_usage: {self.current_device_memory_usage >> 20} MB"  # noqa: E501
        )
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")
        print("Top 10 memory-consuming codes")
        for stat in top_stats[:10]:
            print(stat)
        tracemalloc.start()


garbage_collect_evaluater = GarbageCollectEvaluate()


def garbage_collect():
    global garbage_collect_evaluater
    if garbage_collect_evaluater.is_shoule_collect():
        garbage_collect_evaluater.collect()


def current_location(name=None, stack_depth=-1, print_stack=False):
    stack = traceback.extract_stack()
    if stack_depth < 0:
        stack_depth = len(stack) + stack_depth

    name = name if name else "!"
    for i in range(stack_depth, -1, -1):
        file, line, func, text = stack[i]
        if "op_tools/apply_hook.py" in file or "op_tools/utils.py" in file or "op_tools/base_hook.py" in file:
            stack_depth = i - 1

    for i in range(stack_depth, -1, -1):
        file, line, func, text = stack[i]
        if torch.__path__[0] in file or "/torch_" in file:
            stack_depth = i - 1  # skip internal stack in torch, torch_npu, etc.
        else:
            break

    if print_stack or int(os.getenv("OP_TOOLS_PRINT_STACK", "0")) > 0:
        for i in range(len(stack) - 2):
            file, line, func, text = stack[i]
            print(f"{file}:{line} {func} {text}")

    file, line, func, text = stack[stack_depth]
    return f"{file}:{line} {func}: {text}"


def set_env_if_env_is_empty(env_name, env_value):
    if os.environ.get(env_name, None) is None:
        os.environ[env_name] = env_value
