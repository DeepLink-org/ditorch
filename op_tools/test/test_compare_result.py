# Copyright (c) 2024, DeepLink.
from op_tools.utils import compare_result, tensor_cos_similarity
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
        self.assertTrue(isinstance(compare_info["result_list"], list))

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
        self.assertTrue(isinstance(compare_info["result_list"], list))

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
        self.assertTrue(isinstance(compare_info["result_list"], list))

    def test_compare_same_int_list(self):
        result1 = [t for t in range(10)]
        compare_info = compare_result("same_int_list", result1, result1)
        self.assertTrue(compare_info["allclose"])
        self.assertTrue(compare_info["max_abs_diff"] == 0)
        self.assertTrue(compare_info["max_relative_diff"] == 0)
        self.assertTrue(isinstance(compare_info["result_list"], list))

    def test_compare_diff_int_list(self):
        result1 = [t for t in range(10)]
        result2 = [t * 2 for t in range(10)]
        compare_info = compare_result("diff_int_list", result1, result2)
        self.assertTrue(compare_info["allclose"] is False, compare_info)
        self.assertTrue(compare_info["max_abs_diff"] == 9, compare_info)
        self.assertTrue(compare_info["max_relative_diff"] <= 1)
        self.assertTrue(isinstance(compare_info["result_list"], list))

    def test_same_torch_return_type(self):
        result1 = torch.randn(10, 10).cuda().sort()

        compare_info = compare_result("same_torch_return_type", result1, result1)
        self.assertTrue(compare_info["allclose"] is True)
        self.assertTrue(compare_info["max_abs_diff"] == 0)
        self.assertTrue(isinstance(compare_info["result_list"], list))

    def test_diff_torch_return_type(self):
        result1 = torch.randn(10, 10).cuda().sort()
        result2 = torch.randn(10, 10).cuda().sort()

        compare_info = compare_result("diff_torch_return_type", result1, result2)
        self.assertTrue(compare_info["allclose"] is False)
        self.assertTrue(isinstance(compare_info["result_list"], list))

    def test_compare_same_int(self):
        for i in range(10):
            result1 = i
            result2 = i
            compare_info = compare_result("same_int", result1, result2)
            self.assertTrue(compare_info["allclose"] is True)
            self.assertTrue(compare_info["max_abs_diff"] == 0)
            self.assertTrue(isinstance(compare_info["result_list"], list))

    def test_compare_different_int(self):
        for i in range(1, 10):
            result1 = i
            result2 = i * 2 + 10
            compare_info = compare_result("different_int", result1, result2)
            self.assertTrue(compare_info["allclose"] is False)
            self.assertTrue(compare_info["max_abs_diff"] == i + 10)
            self.assertTrue(compare_info["max_relative_diff"] < (abs(result1 - result2) / result2))
            self.assertTrue(isinstance(compare_info["result_list"], list))

    def test_compare_same_float(self):
        for i in range(10):
            result1 = float(i)
            result2 = float(i)
            compare_info = compare_result("same_float", result1, result2)
            self.assertTrue(compare_info["allclose"] is True)
            self.assertTrue(compare_info["max_abs_diff"] == 0)
            self.assertTrue(compare_info["max_relative_diff"] <= (abs(result1 - result2) / (result2 + 1e-9)))
            self.assertTrue(isinstance(compare_info["result_list"], list))

    def test_compare_different_float(self):
        for i in range(1, 10):
            result1 = float(i)
            result2 = float(i * 2 + 10)
            compare_info = compare_result("different_float", result1, result2)
            self.assertTrue(compare_info["allclose"] is False)
            self.assertTrue(compare_info["max_abs_diff"] == i + 10)
            self.assertTrue(compare_info["max_relative_diff"] < (abs(result1 - result2) / result2))
            self.assertTrue(isinstance(compare_info["result_list"], list))

    def test_compare_same_bool(self):
        for i in range(10):
            result1 = bool(i % 2 == 0)
            result2 = bool(i % 2 == 0)
            compare_info = compare_result("same_bool", result1, result2)
            self.assertTrue(compare_info["allclose"] is True)
            self.assertTrue(compare_info["max_abs_diff"] == 0)
            self.assertTrue(compare_info["max_relative_diff"] == 0)
            self.assertTrue(isinstance(compare_info["result_list"], list))

    def test_compare_different_bool(self):
        for i in range(1, 10):
            result1 = bool(i % 2 == 0)
            result2 = bool(i % 2 == 1)
            compare_info = compare_result("different_bool", result1, result2)
            self.assertTrue(compare_info["allclose"] is False)
            self.assertTrue(math.isnan(compare_info["max_abs_diff"]))
            self.assertTrue(math.isnan(compare_info["max_relative_diff"]))
            self.assertTrue(isinstance(compare_info["result_list"], list))

    def test_compare_empty_tensor(self):
        result1 = torch.empty(0).cuda()
        result2 = torch.empty(0).cuda()
        compare_info = compare_result("empty_tensor", result1, result2)
        self.assertTrue(compare_info["allclose"])

    def test_compare_empty_list(self):
        result1 = []
        result2 = []
        compare_info = compare_result("empty_list", result1, result2)
        self.assertTrue(compare_info["allclose"])

    def test_compare_diff_shape_tensor(self):
        result1 = torch.randn(10, 10).cuda()
        result2 = torch.randn(20, 20).cuda()
        compare_info = compare_result("diff_shape_tensor", result1, result2)
        self.assertFalse(compare_info["allclose"])
        self.assertIn("Inconsistent shape", compare_info["error_info"])

    def test_compare_mixed_types(self):
        result1 = [1, 2.0, 3]
        result2 = [1, 2, 3.0]
        compare_info = compare_result("mixed_types", result1, result2)
        self.assertFalse(compare_info["allclose"])
    
    def test_compare_nested_list(self):
        result1 = [1, 2, 3]
        result2 = [[1,2], 3]
        compare_info = compare_result("nested_list", result1, result2)
        self.assertFalse(compare_info["allclose"])    

    def test_compare_invalid_type(self):
        compare_info = compare_result("invalid_type", {}, [])
        self.assertTrue(compare_info["allclose"])

    def test_compare_invalid_value_a(self):
        result1 = ["1", 2.0, 3]
        result2 = [1, 2, 3.0]
        compare_info = compare_result("invalid_string_a", result1, result2)
        self.assertFalse(compare_info["allclose"])

    def test_compare_invalid_value_b(self):
        result1 = [1, 2.0, 3]
        result2 = ["1", 2, 3.0]
        compare_info = compare_result("invalid_string_b", result1, result2)
        self.assertFalse(compare_info["allclose"])

    def test_compare_same_dict(self):
        result1 = {"1": 1}
        result2 = {"1": 1}
        compare_info = compare_result("same_dict", result1, result2)
        self.assertTrue(compare_info["allclose"])

    def test_compare_different_dict(self):
        result1 = {"1": 2}
        result2 = {"1": 1}
        compare_info = compare_result("different_dict", result1, result2)
        self.assertFalse(compare_info["allclose"])

    def test_compare_different_dict2(self):
        result1 = {"1": 2}
        result2 = {"2": 2}
        compare_info = compare_result("different_dict", result1, result2)
        self.assertFalse(compare_info["allclose"])

    def test_compare_same_dict_list_value(self):
        result1 = {"1": [1, 2, 3]}
        result2 = {"1": [1, 2, 3]}
        compare_info = compare_result("same_dict_list_value", result1, result2)
        self.assertTrue(compare_info["allclose"])

    def test_compare_different_dict_list_value(self):
        result1 = {"1": [2, 4, 6]}
        result2 = {"1": [1, 2, 3]}
        compare_info = compare_result("different_dict_list_value", result1, result2)
        self.assertFalse(compare_info["allclose"])

    def test_compare_dict_different_shape(self):
        result1 = {"1": [2, 4, 6], "2": [4, 5, 6]}
        result2 = {"1": [1, 2, 3]}
        compare_info = compare_result("dict_different_shape", result1, result2)
        self.assertFalse(compare_info["allclose"])

    def test_compare_dict_different_list_shape(self):
        result1 = {"1": [2, 4, 6, 8]}
        result2 = {"1": [1, 2, 3]}
        compare_info = compare_result("dict_different_list_shape", result1, result2)
        self.assertFalse(compare_info["allclose"])

    def test_compare_invalid_input(self):
        self.assertTrue(compare_result("empty_list", [], [])["allclose"])  # 输入空列表
        self.assertTrue(compare_result("empty_tesnsor", torch.empty(0).cuda(), torch.empty(0).cuda())["allclose"])  # 输入空张量
        self.assertTrue(compare_result("equal_tesnsor", torch.ones(1).cuda(), torch.ones(1).cuda())["allclose"])  # 输入相等张量empty
        self.assertFalse(
            compare_result("not_equal_tesnsor", torch.rand(1000).cuda(), -torch.rand(1000).cuda())["allclose"]
        )  # 输入相等张量empty
        self.assertTrue(compare_result("invalid_type", (), [])["allclose"])  # 输入空元组和空列表
        self.assertFalse(compare_result("invalid_value_a", ["1", 2, 3], [1, 2, 3])["allclose"])  # 输入a的元素类型不符合要求
        self.assertFalse(compare_result("invalid_value_b", [1, 2, 3], ["1", 2, 3])["allclose"])  # 输入b的元素类型不符合要求

    def test_cosine_similarity(self):
        x = torch.randn(3, 4, 4, device="cuda").float()
        y = torch.randn(3, 4, 4, device="cuda")
        self.assertTrue(abs(tensor_cos_similarity(x, x) - 1) < 1e-6)
        self.assertTrue(abs(tensor_cos_similarity(x, -x) + 1) < 1e-6)
        xy_cos_similarity = tensor_cos_similarity(x, y)
        self.assertTrue(xy_cos_similarity >= -1 and xy_cos_similarity <= 1)


if __name__ == "__main__":
    unittest.main()
