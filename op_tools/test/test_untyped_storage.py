# Copyright (c) 2024, DeepLink.
import torch
import ditorch
import op_tools

import unittest


class TestOpToolWithSpecialOp(unittest.TestCase):

    def test_untyped_storage(self):
        with op_tools.OpAutoCompare():
            x = torch.randn(3, 4, 5, dtype=torch.float32, device="cuda")
            y = x.untyped_storage()


if __name__ == "__main__":
    unittest.main()
