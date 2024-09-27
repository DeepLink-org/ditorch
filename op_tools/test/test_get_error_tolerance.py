# Copyright (c) 2024, DeepLink.
from op_tools.utils import get_error_tolerance
import os
import unittest
import torch


class TestCustomErrorTolerance(unittest.TestCase):
    def _test_get_error_tolerance(self, dtype, atol, rtol, op_name="test.op_name"):
        atol_, rtol_ = get_error_tolerance(dtype, op_name)
        self.assertEqual(atol, atol_)
        self.assertEqual(rtol, rtol_)
        self.assertTrue(atol > 0)
        self.assertTrue(rtol > 0)
    
    def _tearDown(self):
        # Clean up environment variables to avoid side effects
        env_list = ["AUTOCOMPARE_ERROR_TOLERANCE", "AUTOCOMPARE_ERROR_TOLERANCE_FLOAT32", "AUTOCOMPARE_ERROR_TOLERANCE_FLOAT16", "AUTOCOMPARE_ERROR_TOLERANCE_FLOAT64",
        "OP_NAME_AUTOCOMPARE_ERROR_TOLERANCE_FLOAT16", "OP_NAME_AUTOCOMPARE_ERROR_TOLERANCE_FLOAT32", "OP_NAME_AUTOCOMPARE_ERROR_TOLERANCE_FLOAT64"]
        if "AUTOCOMPARE_ERROR_TOLERANCE" in os.environ:
            del os.environ["AUTOCOMPARE_ERROR_TOLERANCE"]
        for env_name in os.environ:
            del os.environ[env_name]

    def test_default_error_tolerance(self):
        # Testing default tolerances without any environment variables set
        self._test_get_error_tolerance(torch.float16, 1e-4, 1e-4)
        self._test_get_error_tolerance(torch.bfloat16, 1e-3, 1e-3)
        self._test_get_error_tolerance(torch.float32, 1e-5, 1e-5)
        self._test_get_error_tolerance(torch.float64, 1e-8, 1e-8)
        self._tearDown()
    
    def test_environment_variable_override(self):
        os.environ["AUTOCOMPARE_ERROR_TOLERANCE"] = "2,3"
        self._test_get_error_tolerance(torch.float16, 2, 3)
        self._test_get_error_tolerance(torch.float32, 2, 3)

        os.environ["AUTOCOMPARE_ERROR_TOLERANCE_FLOAT32"] = "5e-5,6e-5"
        self._test_get_error_tolerance(torch.float32, 5e-5, 6e-5)
        self._tearDown()

    def test_operation_specific_override(self):
        os.environ["OP_NAME_AUTOCOMPARE_ERROR_TOLERANCE_FLOAT32"] = "30,40"
        self._test_get_error_tolerance(torch.float32, 30, 40, op_name="test.op_name")
        self._tearDown()

    def test_unknown_dtype(self):
        # Test with a dtype that is not explicitly defined
        self._test_get_error_tolerance(torch.int32, 1e-3, 1e-3)
        self._tearDown()

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
        os.environ["OP_NAME_AUTOCOMPARE_ERROR_TOLERANCE_FLOAT32"] = "30,40"
        self._test_get_error_tolerance(torch.float32, 30, 40, op_name="test.op_name")
        self._tearDown()


if __name__ == "__main__":
    unittest.main()
