# Copyright (c) 2024, DeepLink.
import torch
import ditorch
import op_tools
from op_tools.utils import traverse_container
import unittest


class TestOpToolWithSpecialOp(unittest.TestCase):

    def test_untyped_storage(self):
        x = torch.randn(4, 5, dtype=torch.float32, device="cuda")
        y = x.untyped_storage()
        value_list = []
        for item in traverse_container(y):
            value_list.append(item)

        self.assertTrue(len(value_list) == len(y))

        for i in range(len(value_list)):
            self.assertTrue(value_list[i] == y[i])

        with op_tools.OpAutoCompare():
            x = torch.randn(4, 5, dtype=torch.float32, device="cuda")
            y = x.untyped_storage()  # type(y) is class 'torch.storage.UntypedStorage'

    def test_sort(self):
        x = torch.randn(3, 4, 5, dtype=torch.float32, device="cuda")
        y = x.sort()
        sorted_value, indices = y

        value_list = []
        for item in traverse_container(y):
            value_list.append(item)
        self.assertTrue(len(value_list) == 2)

        self.assertTrue(torch.allclose(sorted_value, value_list[0]))
        self.assertTrue(torch.allclose(indices, value_list[1]))

        with op_tools.OpAutoCompare():
            x = torch.randn(3, 4, 5, dtype=torch.float32, device="cuda")
            y = x.sort()  # type(y) is class 'torch.return_types.sort'

    def test_traverse_container_with_dtype(self):
        x = torch.randn(3, 4, 5, dtype=torch.float32, device="cuda")
        y = x.dtype

        value_list = []
        for item in traverse_container(y):
            value_list.append(item)
        self.assertTrue(len(value_list) == 1)
        self.assertTrue(y == value_list[0])

    def test_traverse_container_with_shape(self):
        x = torch.randn(3, 4, 5, dtype=torch.float32, device="cuda")
        y = x.shape

        value_list = []
        for item in traverse_container(y):
            value_list.append(item)
        self.assertTrue(len(value_list) == x.dim())
        for i in range(len(value_list)):
            self.assertTrue(value_list[i] == x.shape[i])


if __name__ == "__main__":
    unittest.main()
