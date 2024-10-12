# Copyright (c) 2024, DeepLink.
from op_tools.utils import (
    is_opname_match,
    is_inplace_op,
    is_view_op,
    get_dtype_cast_dict_form_str,
)
import torch

import unittest


class TestOpNameMatch(unittest.TestCase):

    def test_opname_match(self):
        self.assertEqual(is_opname_match("torch.add", "torch.add"), True)
        self.assertEqual(is_opname_match("torch.add", "torch.mul"), False)
        self.assertEqual(is_opname_match("torch.add", "torch.addc,torch.sub"), False)
        self.assertEqual(is_opname_match("torch.addc", "torch.addc,torch.sub"), True)
        self.assertEqual(is_opname_match("torch.addc", "torch.add,torch.sub"), False)
        self.assertEqual(is_opname_match("torch.sub", "torch.addc,torch.sub"), True)
        self.assertEqual(is_opname_match("torch.sub", "torch.addc,torch.subc,torch.mul"), False)
        self.assertEqual(is_opname_match("torch.subc", "torch.addc,torch.sub,torch.mul"), False)
        self.assertEqual(is_opname_match("torch.subc", ".*"), True)
        self.assertEqual(is_opname_match("torch.subc", "torch.add,.*"), True)
        self.assertEqual(is_opname_match("torch.subc", None), True)
        self.assertEqual(is_opname_match("torch.subc", ""), False)
        self.assertEqual(is_opname_match(None, "torch.add"), False)
        self.assertEqual(is_opname_match(None, None), False)
        self.assertEqual(is_opname_match("torch.add", " torch.add , torch.sub "), True)
        self.assertEqual(is_opname_match("torch.add", " torch.add "), True)

    def test_long_op_list(self):
        op_list = "torch.Tensor.__bool__,torch.Tensor.__rpow__,torch.Tensor.__rtruediv__,torch.Tensor.clone,torch.Tensor.cpu,torch.Tensor.div,torch.Tensor.expand_as,torch.Tensor.fill_,torch.Tensor.float,torch.Tensor.is_complex,torch.Tensor.is_floating_point,torch.Tensor.item,torch.Tensor.mean,torch.Tensor.random_,torch.Tensor.std,torch.Tensor.to,torch.Tensor.view,torch.cat,torch.nn.init.kaiming_uniform_,torch.nn.init.normal_,torch.nn.init.normal_,torch.stack"
        self.assertEqual(is_opname_match("conv2d", op_list), False)
        for op_name in op_list.split(","):
            self.assertEqual(is_opname_match(op_name, op_list), True)

    def test_inplace_op(self):
        self.assertEqual(is_inplace_op("torch.Tensor.add_"), True)
        self.assertEqual(is_inplace_op("torch.Tensadd"), False)
        self.assertEqual(is_inplace_op("torch.Tensor.__getitem__"), False)
        self.assertEqual(is_inplace_op("torch.Tensor.__setitem__"), True)

    def test_view_op(self):
        self.assertEqual(is_view_op("torch.Tensor.add_"), False)
        self.assertEqual(is_view_op("torch.Tensadd"), False)
        self.assertEqual(is_view_op("torch.Tensor.view"), True)

    def test_get_dtype_cast_dict_from_config(self):
        dtype_cast_dict = get_dtype_cast_dict_form_str("torch.float32->torch.float16,torch.float64->torch.float16,torch.int64->torch.int32")
        self.assertEqual(
            dtype_cast_dict,
            {
                torch.float32: torch.float16,
                torch.float64: torch.float16,
                torch.int64: torch.int32,
            },
        )

    def test_get_dtype_cast_dict_from_config2(self):
        dtype_cast_dict = get_dtype_cast_dict_form_str(" torch.float32 ->torch.float16, torch.float64 -> torch.float16, torch.int64->  torch.int32 ")
        self.assertEqual(
            dtype_cast_dict,
            {
                torch.float32: torch.float16,
                torch.float64: torch.float16,
                torch.int64: torch.int32,
            },
        )


if __name__ == "__main__":
    unittest.main()
