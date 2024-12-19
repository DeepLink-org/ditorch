# Copyright (c) 2024, DeepLink.

import unittest
import torch
from ditorch.utils import is_to_fp32_tensor


# @is_to_fp32_tensor(to_fp32=True)
def test_func_to_fp32(tensor: torch.Tensor, tensors_list, tensors_dict):
    assert (
        tensor.dtype == torch.float32
    ), f"tensor's dtype is not fp32, but {tensor.dtype}"
    for tensor in tensors_list:
        if isinstance(tensor, torch.Tensor):
            assert (
                tensor.dtype == torch.float32
            ), f"tensor's dtype is not fp32, but {tensor.dtype}"
    for k, v in tensors_dict.items():
        if isinstance(v, torch.Tensor):
            assert (
                v.dtype == torch.float32
            ), f"tensor's dtype is not fp32, but {v.dtype}"


test_func_to_fp321 = is_to_fp32_tensor(to_fp32=True)(test_func_to_fp32)


class TestUtils(unittest.TestCase):
    def test_is_to_fp32_tensor(self):
        tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
        tensors_list = [
            torch.tensor([1.0, 1.0, 1.0], dtype=torch.float16),
            torch.tensor([2.0, 2.0, 2.0], dtype=torch.float16),
            torch.tensor([3.0, 3.0, 3.0], dtype=torch.float16),
        ]
        tensors_dict = {
            "tensor1": torch.tensor([1.0, 1.0, 1.0], dtype=torch.float16),
            "tensor2": torch.tensor([2.0, 2.0, 2.0], dtype=torch.float16),
            "tensor3": torch.tensor([3.0, 3.0, 3.0], dtype=torch.float16),
        }
        test_func_to_fp321(tensor, tensors_list, tensors_dict)


if __name__ == '__main__':
    unittest.main()
