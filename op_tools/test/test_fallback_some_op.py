# Copyright (c) 2024, DeepLink.
import torch
import op_tools
import unittest


class TestFallbackOp(unittest.TestCase):

    def test_fallback_op(self):
        return
        op_tools.fallback_ops(ops=["torch.add", "torch.mul"])
        op_tools.fallback_ops(torch.sub)
        x = torch.tensor([1, 2, 3], device="cuda")
        y = torch.tensor([4, 5, 6], device="cuda")
        z = torch.add(x, y)
        assert z.is_cpu == False

        n = torch.sub(x, y)
        assert n.is_cpu == False

        m = torch.mul(x, y)
        assert m.is_cpu == False

    def test_dump_all_args(self):
        op_tools.dump_all_ops_args()
        x = torch.tensor([1, 2, 3], device="cuda")
        y = torch.tensor([4, 5, 6], device="cuda")

        z = torch.add(x, y)


if __name__ == "__main__":
    unittest.main()
