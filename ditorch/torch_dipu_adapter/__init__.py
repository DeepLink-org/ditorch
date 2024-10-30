# Copyright (c) 2024, DeepLink.
import torch  # noqa: F401
import torch_dipu  # noqa: F401


def mock():
    if not hasattr(torch_dipu, "__version__"):
        torch_dipu.__version__ = torch.__version__


framework = torch_dipu
arch = torch_dipu.vendor_type
