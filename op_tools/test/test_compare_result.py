# Copyright (c) 2024, DeepLink.
from op_tools.utils import compare_result
import torch
import ditorch
import unittest
import math


class TestCompareResult(unittest.TestCase):

    def test_compare_same_tensor(self):
        result1 = torch.randn(10, 10).cuda()
        compare_info = compare_result("same_tensor", result1, result1)
        self.assertTrue(compare_info["allclose"])
        self.assertTrue(compare_info["max_abs_diff"] == 0)
        self.assertTrue(compare_info["max_relative_diff"] == 0)
        self.assertTrue(compare_info["error_info"] == "")

    def test_compare_randn_tensor(self):
        result1 = torch.randn(10, 10).cuda()
        result2 = torch.randn(10, 10).cuda()
        max_abs = torch.abs(result1 - result2)
        max_abs_diff = torch.max(max_abs).item()
        max_relate_diff = (max_abs / (torch.abs(result1) + 1e-6)).max().item()
        compare_info = compare_result("randn_tensor", result1, result2)
        self.assertTrue(compare_info["allclose"] is False)
        self.assertTrue(compare_info["max_abs_diff"] == max_abs_diff)
        self.assertTrue(
            abs(compare_info["max_relative_diff"] - max_relate_diff) < 1e-3,
            f"{compare_info['max_relative_diff']} != {max_relate_diff}",
        )
        self.assertTrue(compare_info["error_info"] == "")

    def test_compare_randn_tensor_list(self):
        result1 = torch.randn(10, 10).cuda()
        result2 = torch.randn(10, 10).cuda()
        max_abs = torch.abs(result1 - result2)
        max_abs_diff = torch.max(max_abs).item()
        max_relate_diff = (max_abs / (torch.abs(result1) + 1e-6)).max().item()

        tensor_list1 = [result1, result1]
        tensor_list2 = [result2, result2]

        compare_info = compare_result("randn_tensor_list", tensor_list1, tensor_list2)
        self.assertTrue(compare_info["allclose"] is False)
        self.assertTrue(compare_info["max_abs_diff"] == max_abs_diff)
        self.assertTrue(
            abs(compare_info["max_relative_diff"] - max_relate_diff) < 1e-3,
            f"{compare_info['max_relative_diff']} != {max_relate_diff}",
        )
        self.assertTrue(compare_info["error_info"] == "")

    def test_compare_same_int_list(self):
        result1 = [t for t in range(10)]
        compare_info = compare_result("same_int_list", result1, result1)
        self.assertTrue(compare_info["allclose"])
        self.assertTrue(compare_info["max_abs_diff"] == 0)
        self.assertTrue(compare_info["max_relative_diff"] == 0)

    def test_compare_diff_int_list(self):
        result1 = [t for t in range(10)]
        result2 = [t * 2 for t in range(10)]
        compare_info = compare_result("diff_int_list", result1, result2)
        self.assertTrue(compare_info["allclose"] is False, compare_info)
        self.assertTrue(compare_info["max_abs_diff"] == 9, compare_info)
        self.assertTrue(abs(compare_info["max_relative_diff"] - 1) < 1e-3, compare_info)

    def test_same_torch_return_type(self):
        result1 = torch.randn(10, 10).cuda().sort()

        compare_info = compare_result("same_torch_return_type", result1, result1)
        self.assertTrue(compare_info["allclose"] is True)
        self.assertTrue(compare_info["max_abs_diff"] == 0)

    def test_diff_torch_return_type(self):
        result1 = torch.randn(10, 10).cuda().sort()
        result2 = torch.randn(10, 10).cuda().sort()

        compare_info = compare_result("diff_torch_return_type", result1, result2)
        self.assertTrue(compare_info["allclose"] is False)

    def test_compare_same_int(self):
        for i in range(10):
            result1 = i
            result2 = i
            compare_info = compare_result("same_int", result1, result2)
            self.assertTrue(compare_info["allclose"] is True)
            self.assertTrue(compare_info["max_abs_diff"] == 0)

    def test_compare_different_int(self):
        for i in range(1, 10):
            result1 = i
            result2 = i * 2 + 10
            compare_info = compare_result("different_int", result1, result2)
            self.assertTrue(compare_info["allclose"] is False)
            self.assertTrue(compare_info["max_abs_diff"] == i + 10)
            self.assertTrue(
                abs(compare_info["max_relative_diff"] - ((i + 10) / i)) < 1e-3
            )

    def test_compare_same_float(self):
        for i in range(10):
            result1 = float(i)
            result2 = float(i)
            compare_info = compare_result("same_float", result1, result2)
            self.assertTrue(compare_info["allclose"] is True)
            self.assertTrue(compare_info["max_abs_diff"] == 0)
            self.assertTrue(abs(compare_info["max_relative_diff"] - 0) < 1e-3)

    def test_compare_different_float(self):
        for i in range(1, 10):
            result1 = float(i)
            result2 = float(i * 2 + 10)
            compare_info = compare_result("different_float", result1, result2)
            self.assertTrue(compare_info["allclose"] is False)
            self.assertTrue(compare_info["max_abs_diff"] == i + 10)
            self.assertTrue(
                abs(compare_info["max_relative_diff"] - ((i + 10) / i)) < 1e-3
            )

    def test_compare_same_bool(self):
        for i in range(10):
            result1 = bool(i % 2 == 0)
            result2 = bool(i % 2 == 0)
            compare_info = compare_result("same_bool", result1, result2)
            self.assertTrue(compare_info["allclose"] is True)
            self.assertTrue(compare_info["max_abs_diff"] == 0)
            self.assertTrue(compare_info["max_relative_diff"] == 0)

    def test_compare_different_bool(self):
        for i in range(1, 10):
            result1 = bool(i % 2 == 0)
            result2 = bool(i % 2 == 1)
            compare_info = compare_result("different_bool", result1, result2)
            self.assertTrue(compare_info["allclose"] is False)
            self.assertTrue(math.isnan(compare_info["max_abs_diff"]))
            self.assertTrue(math.isnan(compare_info["max_relative_diff"]))


if __name__ == "__main__":
    unittest.main()
