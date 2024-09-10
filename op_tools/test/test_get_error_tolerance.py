# Copyright (c) 2024, DeepLink.
from op_tools.utils import get_error_tolerance
import os
import unittest
import torch


class TestCustomErrorTolerance(unittest.TestCase):
    def _test_get_error_tolerance(self, dtype, atol, rtol):
        atol_, rtol_ = get_error_tolerance(dtype)
        self.assertEqual(atol, atol_)
        self.assertEqual(rtol, rtol_)
        self.assertTrue(atol > 0)
        self.assertTrue(rtol > 0)

    def test_get_error_tolerance(self):
        os.environ["AUTOCOMPARE_ERROR_TOLERANCE"] = "2,3"
        self._test_get_error_tolerance(torch.float16, 2, 3)
        self._test_get_error_tolerance(torch.float32, 2, 3)
        self._test_get_error_tolerance(torch.float64, 2, 3)
        self._test_get_error_tolerance(torch.int, 2, 3)
        os.environ["AUTOCOMPARE_ERROR_TOLERANCE_FLOAT16"] = "1e-8,2e-9"
        self._test_get_error_tolerance(torch.float16, 1e-8, 2e-9)
        os.environ["AUTOCOMPARE_ERROR_TOLERANCE_FLOAT32"] = "2e-8,3e-9"
        self._test_get_error_tolerance(torch.float32, 2e-8, 3e-9)
        os.environ["AUTOCOMPARE_ERROR_TOLERANCE_FLOAT64"] = "3e-8,4e-9"
        self._test_get_error_tolerance(torch.float64, 3e-8, 4e-9)
        self._test_get_error_tolerance(torch.int32, 2, 3)


if __name__ == "__main__":
    unittest.main()
