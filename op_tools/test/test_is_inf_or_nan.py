# Copyright (c) 2024, DeepLink.
from op_tools.utils import is_inf_or_nan
import torch

import unittest


class TestInfOrNan(unittest.TestCase):

    def test_is_inf_or_nan(self):
        self.assertEqual(is_inf_or_nan(torch.tensor(float("inf"))), True)
        self.assertEqual(is_inf_or_nan(torch.tensor(float("+inf"))), True)
        self.assertEqual(is_inf_or_nan(torch.tensor(float("-inf"))), True)
        self.assertEqual(is_inf_or_nan(float("inf")), True)
        self.assertEqual(is_inf_or_nan(float("+inf")), True)
        self.assertEqual(is_inf_or_nan(float("-inf")), True)
        self.assertEqual(is_inf_or_nan([float("-inf")]), [True])
        self.assertEqual(is_inf_or_nan([float("-inf"), 0]), [True, False])
        self.assertEqual(is_inf_or_nan([float("-inf"), torch.tensor(float(1))]), [True, False])
        self.assertEqual(is_inf_or_nan(torch.add(torch.randn(3, 4), torch.randn(3, 4))), False)
        self.assertEqual(any(is_inf_or_nan(torch.sort(torch.randn(3, 4)))), False)
        self.assertEqual(
            is_inf_or_nan(torch.sort(torch.randn(3, 4))),
            (
                False,
                False,
            ),
        )


if __name__ == "__main__":
    unittest.main()
