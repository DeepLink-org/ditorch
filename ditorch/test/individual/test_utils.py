# Copyright (c) 2024, DeepLink.

import unittest
import torch
from ditorch.utils import is_to_fp32_tensor, tensor_op_inp


@is_to_fp32_tensor(to_fp32=True)
def to_fp32_and_add_1(tensor: torch.Tensor, tensors_list, tensors_dict):
    assert (
        tensor.dtype == torch.float32
    ), f"tensor's dtype is not fp32, but {tensor.dtype}"
    tensor.add_(1)
    for tensor in tensors_list:
        if isinstance(tensor, torch.Tensor):
            assert (
                tensor.dtype == torch.float32
            ), f"tensor's dtype is not fp32, but {tensor.dtype}"
        tensor.add_(1)
    for k, v in tensors_dict.items():
        if isinstance(v, torch.Tensor):
            assert (
                v.dtype == torch.float32
            ), f"tensor's dtype is not fp32, but {v.dtype}"
        v.add_(1)


def test_is_to_fp32_tensor():
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
    to_fp32_and_add_1(tensor, tensors_list, tensors_dict)
    expected_tensor = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float16)
    expected_tensors_list = [
        torch.tensor([2.0, 2.0, 2.0], dtype=torch.float16),
        torch.tensor([3.0, 3.0, 3.0], dtype=torch.float16),
        torch.tensor([4.0, 4.0, 4.0], dtype=torch.float16),
    ]
    expected_tensors_dict = {
        "tensor1": torch.tensor([2.0, 2.0, 2.0], dtype=torch.float16),
        "tensor2": torch.tensor([3.0, 3.0, 3.0], dtype=torch.float16),
        "tensor3": torch.tensor([4.0, 4.0, 4.0], dtype=torch.float16),
    }

    assert (
        tensor.dtype == expected_tensor.dtype
    ), f"tensor's dtype ({tensor.dtype}) is not the same as expected_tensor's ({expected_tensor.dtype})"
    assert torch.allclose(
        tensor, expected_tensor
    ), f"tensor's value ({tensor}) is not the same as expected_tensor's {expected_tensor}."

    for i in range(len(tensors_list)):
        assert (
            tensors_list[i].dtype == expected_tensors_list[i].dtype
        ), f"tensor's dtype ({tensors_list[i].dtype}) is not the same as expected_tensor's ({expected_tensors_list[i].dtype})"
        assert torch.allclose(
            tensors_list[i], expected_tensors_list[i]
        ), f"tensor's value ({tensors_list[i]}) is not the same as expected_tensor's {expected_tensors_list[i]}."

    for k, v in tensors_dict.items():
        assert (
            v.dtype == expected_tensors_dict[k].dtype
        ), f"tensor's dtype ({v.dtype}) is not the same as expected_tensor's ({expected_tensors_dict[k].dtype})"
        assert torch.allclose(
            v, expected_tensors_dict[k]
        ), f"tensor's value ({v}) is not the same as expected_tensor's {expected_tensors_dict[k]}."


def test_tensor_op_inp():
    tensor = torch.tensor([3.0, 3.0, 3.0], dtype=torch.float32)
    tensor_op_inp(tensor, "div_", 2.0)
    tensor_expected = torch.Tensor([1.5, 1.5, 1.5])
    assert torch.allclose(tensor, tensor_expected), f"tensor's value ({tensor}) is not the same as expected_tensor's {tensor_expected}."

    para = torch.nn.Parameter(torch.tensor([3.0, 3.0, 3.0], dtype=torch.float32))
    tensor_op_inp(para, "div_", 2.0)
    tensor_expected = torch.Tensor([1.5, 1.5, 1.5])
    assert torch.allclose(para, tensor_expected), f"tensor's value ({para}) is not the same as expected_tensor's {tensor_expected}."


class TestUtils(unittest.TestCase):
    def test_is_to_fp32_tensor(self):
        test_is_to_fp32_tensor()

    def test_tensor_op_inp(self):
        test_tensor_op_inp()


if __name__ == "__main__":
    unittest.main()
