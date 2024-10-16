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

        with op_tools.OpTimeMeasure():
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

        with op_tools.OpTimeMeasure():
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

    def test_overflow(self):
        x = torch.randn(3, 4, 5, dtype=torch.float32, device="cuda")
        y = torch.zeros_like(x)
        z = x / y
        # print(z)  # all elem is inf or -inf

        nan_detect_result = torch.isnan(z)
        assert not nan_detect_result.any().item()
        # print(nan_detect_result) # all elem is False

        inf_detect_result = torch.isinf(z)
        assert inf_detect_result.all().item()
        # print(inf_detect_result) # all elem is True

        # torch.isfinite
        finite_detect_result = torch.isfinite(z)
        assert not finite_detect_result.any().item()

    def test_overflow2(self):
        x = torch.full((3, 4, 5,), dtype=torch.float32, device="cuda", fill_value=3.402823466e38)
        self.assertFalse(torch.isinf(x).any().item())
        print(x)
        y = x + x
        print(y)
        self.assertTrue(torch.isinf(y).all().item())

    def test_overflow3(self):
        fill_value = 3.402823466e38
        x = torch.full((3, 4, 5,), dtype=torch.float32, device="cuda", fill_value=fill_value)
        y = x + 1e-9
        y_cpu = fill_value + 1e-9
        max_value_item = y.max().item()
        # bad results
        error_info = f"Add large numbers to small numbers: {max_value_item} != {y_cpu} + 1e-9, {max_value_item - fill_value} != 1e-9"
        # self.assertTrue(max_value_item - fill_value <= 1e-9, error_info)
        if max_value_item - fill_value > 1e-9:
            print(error_info)
        self.assertFalse(torch.isinf(y).any().item())

    def test_overflow4(self):
        with op_tools.OpOverflowCheck():
            x = torch.randn(3, 4, 5, dtype=torch.float32, device="cuda", requires_grad=True)
            y = torch.zeros_like(x)
            z = x / y
            z.backward(torch.ones_like(z))
            x = torch.full((3, 4, 5,), dtype=torch.float32, device="cuda", fill_value=3.402823466e38)
            y = x + x
            z = x * x

    def test_setitem(self):
        with op_tools.OpAutoCompare():
            x = torch.randn(3, 4, 5, dtype=torch.float32, device="cuda")
            x[0, 0, 0] = 1.0  # __setitem__  return None

        with op_tools.OpTimeMeasure():
            x = torch.randn(3, 4, 5, dtype=torch.float32, device="cuda")
            x[0, 0, 0] = 1.0  # __setitem__  return None

    def test_inplace_op(self):
        def f():
            m = torch.randn(3, 4, 5, dtype=torch.float32, device="cuda", requires_grad=True)
            x = m + 1
            x.add_(1.0)
            x.mul_(2.0)
            x.sub_(1.0)
            x.div_(2.0)
            x.pow_(2.0)
            x.sqrt_()
            y = x.abs()
            y.backward(torch.ones_like(x))

        with op_tools.OpAutoCompare():
            f()

        with op_tools.OpTimeMeasure():
            f()

    def test_contiguous(self):
        def f():
            x = torch.randn(3, 4, 5, dtype=torch.float32, device="cuda", requires_grad=True)
            y = x.contiguous()
            z = y + y
            z.backward(torch.ones_like(z))

            x = torch.randn(3, 4, 5, dtype=torch.float64, device="cuda", requires_grad=True)
            y = x.contiguous()
            z = y + y
            z.backward(torch.ones_like(z))

            x = torch.randn(3, 3, dtype=torch.float64, device="cuda", requires_grad=True)
            x = torch.as_strided(input=z, size=(2, 2), stride=(1, 2))
            y = x.contiguous()
            z = y + y
            z.backward(torch.ones_like(z))

        with op_tools.OpAutoCompare():
            f()

        with op_tools.OpTimeMeasure():
            f()

    def test_exp(self):
        with op_tools.OpAutoCompare():
            x = torch.randn(3, 4, 5, dtype=torch.float16, device="cuda", requires_grad=True)
            y = x.exp()
            y.backward(torch.ones_like(y))

    def test_dtype_cast(self):
        with op_tools.OpDtypeCast():
            x = torch.randn(3, 4, 5, dtype=torch.float16, device="cuda", requires_grad=True)
            y = x.to(torch.float32)
            y.backward(torch.ones_like(y))


if __name__ == "__main__":
    unittest.main()
