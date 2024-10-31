import torch_mlu
from .mock_runtime import mock_runtime


def mock():
    from torch_mlu.utils.model_transfer import transfer  # noqa: F401

    mock_runtime()


framework = torch_mlu
arch = "camb"
