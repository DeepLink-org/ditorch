# Copyright (c) 2024, DeepLink.
import os

framework = None
try:
    from ditorch import torch_npu_adapter

    framework = "torch_npu:" + torch_npu_adapter.torch_npu.__version__
except Exception as e:  # noqa: F841
    pass
try:
    from ditorch import torch_dipu_adapter  # noqa: F401

    framework = "torch_dipu"  # torch_dipu has not __version__ attr

except Exception as e:  # noqa: F841
    pass

try:
    from ditorch import torch_mlu_adapter

    framework = "torch_mlu:" + torch_mlu_adapter.torch_mlu.__version__
except Exception as e:  # noqa: F841
    pass

try:
    from ditorch import torch_biren_adapter

    framework = "torch_br:" + torch_biren_adapter.torch_br.__version__
except Exception as e:  # noqa: F841
    pass


from ditorch import common_adapter  # noqa: F401,E402

print(f"ditorch.framework: {framework}  pid: {os.getpid()}")
