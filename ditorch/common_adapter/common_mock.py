import torch
import os


def mock_tensor_device():
    if torch.__version__ < "2.0.0":
        return
    from torch.overrides import TorchFunctionMode, resolve_name

    class DeviceMock(TorchFunctionMode):
        def __init__(self):
            super().__init__()

        def __torch_function__(self, func, types, args, kwargs=None):
            try:
                name = resolve_name(func)
            except Exception:
                name = None
            result = func(*args, **(kwargs or {}))
            if name == "torch.Tensor.device.__get__":
                if result.type not in ["cpu", "mps", "xpu", "xla", "meta"]:
                    device_str = "cuda"
                    if result.index is not None:
                        device_str += f":{result.index}"
                    result = torch.device(device_str)
            if name == "torch.Tensor.__repr__":
                device = args[0].device
                if device.type != "cpu":
                    result = result.replace(device.type, "cuda")

            return result

    device_mock = DeviceMock()
    if os.getenv("DITORCH_SHOW_DEVICE_AS_CUDA", "1") == "1":
        device_mock.__enter__()
