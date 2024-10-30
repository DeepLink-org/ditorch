# Copyright (c) 2024, DeepLink.
import torch_npu
from .mock_runtime import mock_runtime
from .mock_dist import mock_dist


def mock():
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

    mock_runtime()
    mock_dist()


framework = torch_npu
arch = "ascend"
