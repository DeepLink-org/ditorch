# Copyright (c) 2024, DeepLink.
from op_tools.utils import get_function_from_string

import unittest
import inspect


class TestFunctionFromString(unittest.TestCase):

    def test_string2func(self):
        op_list = "torch.nn.functional.conv2d,torch.add,torch.Tensor.add,torch.Tensor.__bool__,torch.Tensor.__rpow__,torch.Tensor.__rtruediv__,torch.Tensor.clone,torch.Tensor.cpu,torch.Tensor.div,torch.Tensor.expand_as,torch.Tensor.fill_,torch.Tensor.float,torch.Tensor.is_complex,torch.Tensor.is_floating_point,torch.Tensor.item,torch.Tensor.mean,torch.Tensor.random_,torch.Tensor.std,torch.Tensor.to,torch.Tensor.view,torch.cat,torch.nn.init.kaiming_uniform_,torch.nn.init.normal_,torch.nn.init.normal_,torch.stack"
        for op_name in op_list.split(","):
            func = get_function_from_string(op_name)
            self.assertTrue(inspect.isroutine(func))


if __name__ == "__main__":
    unittest.main()
