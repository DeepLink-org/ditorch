import torch
import os

if torch.__version__ >= "2.0.0":
    from torch.overrides import TorchFunctionMode, resolve_name

    class DeviceMock(TorchFunctionMode):
        def __init__(self):
            super().__init__()

        def __torch_function__(self, func, types, args, kwargs=None):
            name = resolve_name(func)
            result = func(*args, **(kwargs or {}))
            if name == "torch.Tensor.device.__get__":
                if result.type != "cpu":
                    result = torch.device("cuda" + (":" + str(result.index)) if result.index is not None else "")
            if name == "torch.Tensor.__repr__":
                device = args[0].device
                if device.type != "cpu":
                    result = result.replace(device.type, "cuda")

            return result

    device_mock = DeviceMock()
    if os.getenv("DITORCH_SHOW_DEVICE_AS_CUDA", "1") == "1":
        device_mock.__enter__()
