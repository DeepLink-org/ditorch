from op_tools.utils import is_opname_match

import unittest


class TestOpNameMatch(unittest.TestCase):

    def test_opname_match(self):
        self.assertEqual(is_opname_match("torch.add", "torch.addc,torch.sub"), False)
        self.assertEqual(is_opname_match("torch.addc", "torch.addc,torch.sub"), True)
        self.assertEqual(is_opname_match("torch.addc", "torch.add,torch.sub"), False)
        self.assertEqual(is_opname_match("torch.sub", "torch.addc,torch.sub"), True)
        self.assertEqual(
            is_opname_match("torch.sub", "torch.addc,torch.subc,torch.mul"), False
        )
        self.assertEqual(
            is_opname_match("torch.subc", "torch.addc,torch.sub,torch.mul"), False
        )
        self.assertEqual(is_opname_match("torch.subc", ".*"), True)
        self.assertEqual(is_opname_match("torch.subc", "torch.add,.*"), True)
        self.assertEqual(is_opname_match("torch.subc", None), True)
        self.assertEqual(is_opname_match("torch.subc", ""), False)

    def test_long_op_list(self):
        op_list = "torch.Tensor.__bool__,torch.Tensor.__rpow__,torch.Tensor.__rtruediv__,torch.Tensor.clone,torch.Tensor.cpu,torch.Tensor.div,torch.Tensor.expand_as,torch.Tensor.fill_,torch.Tensor.float,torch.Tensor.is_complex,torch.Tensor.is_floating_point,torch.Tensor.item,torch.Tensor.mean,torch.Tensor.random_,torch.Tensor.std,torch.Tensor.to,torch.Tensor.view,torch.cat,torch.nn.init.kaiming_uniform_,torch.nn.init.normal_,torch.nn.init.normal_,torch.stack"
        self.assertEqual(is_opname_match("conv2d", op_list), False)
        for op_name in op_list.split(","):
            self.assertEqual(is_opname_match(op_name, op_list), True)


if __name__ == "__main__":
    unittest.main()
