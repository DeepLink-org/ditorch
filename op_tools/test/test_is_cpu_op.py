# Copyright (c) 2024, DeepLink.
from op_tools.utils import is_cpu_op
import torch
import ditorch

import unittest


class TestIsCpuOp(unittest.TestCase):

    def test_is_cpu_op(self):
        self.assertEqual(is_cpu_op(torch.tensor([1, 2, 3])), (True, torch.device("cpu"),))
        self.assertFalse(is_cpu_op(torch.tensor([1, 2, 3], device='cuda'))[0])
        self.assertFalse(is_cpu_op(torch.tensor([1, 2, 3], device='cuda'))[0])
        self.assertTrue(is_cpu_op(torch.tensor([1, 2, 3], dtype=torch.float16))[0])
        self.assertTrue(is_cpu_op(torch.tensor([1, 2, 3], dtype=torch.float16), device='cpu')[0])
        self.assertFalse(is_cpu_op(torch.tensor([1, 2, 3], dtype=torch.float16), device='cuda')[0])


if __name__ == "__main__":
    unittest.main()
