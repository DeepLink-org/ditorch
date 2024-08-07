import torch
from .base_hook import BaseHook, DisableHookGuard
from .utils import to_device, is_cpu_op
from .op_fallback_hook import OpFallbackHook
from .save_op_args import save_op_args


def compre_obj(a, b):
    # We assume they are of the same type: torch.nn.parameter.Parameter and torch.Tensor
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        atol, rtol = 1e-3, 1e-3
        a_cpu, b_cpu = a.cpu(), b.cpu()
        if a_cpu.dtype == torch.bool:
            a_cpu = a_cpu.int()
        if b_cpu.dtype == torch.bool:
            b_cpu = b_cpu.int()
        diff = torch.abs(a_cpu - b_cpu)
        max_diff = diff.max().item()
        if a_cpu.dtype == b_cpu.dtype:
            return max_diff
        else:
            return f"Inconsistent dtypes: {a.dtype} {b.dtype}, max_diff:{max_diff}"
    elif type(a) is not type(b):
        return f"Inconsistent types: {type(a)} {type(b)}"
    elif isinstance(a, (list, tuple)):
        return type(a)([compre_obj(a[i], b[i]) for i in range(len(a))])
    elif isinstance(a, dict):
        return {k: compre_obj(a[k], b[k]) for k in a.keys()}
    elif isinstance(a, (int, float, complex)):
        return abs(a - b)
    elif a is None:
        return 0.0
    else:
        return f"unhandle type:{a} {b}"


def cpu_fp16_to_fp32(obj):
    if isinstance(obj, torch.Tensor):
        if obj.dtype in [torch.half, torch.float16, torch.bfloat16]:
            return obj.to(torch.float32)
        else:
            return obj
    elif isinstance(obj, (tuple, list)):
        return type(obj)([cpu_fp16_to_fp32(v) for v in obj])
    elif isinstance(obj, dict):
        return {k: cpu_fp16_to_fp32(v) for k, v in obj.items()}
    else:
        return obj


def dtype_convert(src, dst):
    if isinstance(src, torch.Tensor):
        if src.dtype == torch.float32 and dst.dtype in [torch.half, torch.float16, torch.bfloat16]:
            return src.to(dst.dtype)
        else:
            return src
    elif isinstance(src, (tuple, list)):
        return type(src)([dtype_convert(src[i], dst[i]) for i in range(len(src))])
    elif isinstance(src, dict):
        return {k: dtype_convert(src[k], dst[k]) for k in src.keys()}
    else:
        return src

class OpAutoCompareHook(BaseHook):
    def __init__(self, name) -> None:
        super().__init__(name)

    def compare_result(self, device_result, cpu_result):
        self.compare_result = compre_obj(device_result, cpu_result)
        allclose = True
        if isinstance(self.compare_result, (int, float, complex)):  # f"{max_diff:.9f}"
            print(
                f"OpAutoCompareHook: {self.name:<50} max_diff: {f'{self.compare_result:20.9f}'}"
            )
            if self.compare_result > 1e-3:
                allclose = False
        elif isinstance(self.compare_result, (list, tuple)):
            for i in range(len(self.compare_result)):
                print(
                    f"OpAutoCompareHook: {self.name:<50} {i}th \tmax_diff: {f'{self.compare_result[i]:20.9f}'}"
                )
                if self.compare_result[i] > 1e-3:
                    allclose = False
        elif isinstance(self.compare_result, (dict,)):
            for k, v in self.compare_result.items():
                print(
                    f"OpAutoCompareHook: {self.name:<50} {k} \tmax_diff: {f'{v:20.9f}'}"
                )
                if v > 1e-3:
                    allclose = False
        else:
            print(
                f"OpAutoCompareHook: {self.name:<50} compare_result: {self.compare_result}"
            )

        if not allclose:
            save_op_args(self.name, "device_input", self.args, self.kwargs)
            save_op_args(self.name, "device_output", self.result)
            save_op_args(self.name, "cpu_input", self.args_cpu, self.kwargs_cpu)
            save_op_args(self.name, "cpu_output", self.result_cpu)

    def before_call_op(self, *args, **kwargs):
        with DisableHookGuard():
            self.is_cpu_op, self.device = is_cpu_op(*args, **kwargs)
            if self.is_cpu_op:
                return
            self.args_cpu = to_device("cpu", self.args)
            self.kwargs_cpu = to_device("cpu", self.kwargs or {})

    def after_call_op(self, result):
        if self.is_cpu_op:
            return
        with DisableHookGuard():
            # convert dtype from float16 or bfloat16 to float32, incase the caculate is not supported on cpu
            self.args_cpu_to_fp32 = cpu_fp16_to_fp32(self.args_cpu)
            self.kwargs_cpu_to_fp32 = cpu_fp16_to_fp32(self.kwargs_cpu)
            self.result_cpu_to_fp32 = self.func(*self.args_cpu_to_fp32, **self.kwargs_cpu_to_fp32)
            # convert the dtype of result back
            self.result_cpu = dtype_convert(self.result_cpu_to_fp32, self.result)
            self.compare_result(self.result, self.result_cpu)
