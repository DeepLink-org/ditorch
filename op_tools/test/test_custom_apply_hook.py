# Copyright (c) 2024, DeepLink.
import torch
import op_tools
import unittest


def _test_function(x, y):
    a = torch.add(x, y) * 2
    b = torch.sub(a, y) / 3
    c = torch.mul(b, a) + 1
    d = torch.div(c, b) - 2
    d.backward(torch.ones_like(d))
    a.is_cpu == x.is_cpu
    b.is_cpu == x.is_cpu
    c.is_cpu == x.is_cpu
    d.is_cpu == x.is_cpu

    a.requires_grad == x.requires_grad
    b.requires_grad == x.requires_grad
    c.requires_grad == x.requires_grad
    d.requires_grad == x.requires_grad
    a.device == x.device
    b.device == x.device
    c.device == x.device
    d.device == x.device
    a.dtype == x.dtype
    b.dtype == x.dtype
    c.dtype == x.dtype
    d.dtype == x.dtype

    a.shape == x.shape
    b.shape == x.shape
    c.shape == x.shape
    d.shape == x.shape

    assert a.grad is None
    assert b.grad is None
    assert c.grad is None
    assert d.grad is None
    assert a.is_leaf is False
    assert b.is_leaf is False
    assert c.is_leaf is False
    assert d.is_leaf is False

    assert (x.grad is not None) == x.requires_grad
    assert (y.grad is not None) == y.requires_grad


class TestCustomApplyHook(unittest.TestCase):
    def test_fallback_op(self):
        op_tools.apply_feature(
            ops=["torch.add", "torch.sub", "torch.mul", "torch.div"], feature="fallback"
        )
        x = torch.tensor(
            [1, 2, 3], dtype=torch.float16, device="cuda", requires_grad=True
        )
        y = torch.tensor(
            [4, 5, 6], dtype=torch.float16, device="cuda", requires_grad=True
        )
        _test_function(x, y)

    def test_dump_all_args(self):
        op_tools.apply_feature(
            ops=["torch.add", "torch.sub", "torch.mul", "torch.div"],
            feature="dump_op_args",
        )
        x = torch.tensor(
            [1, 2, 3], dtype=torch.float16, device="cuda", requires_grad=True
        )
        y = torch.tensor(
            [4, 5, 6], dtype=torch.float16, device="cuda", requires_grad=True
        )

        _test_function(x, y)

    def test_op_capture(self):
        op_tools.apply_feature(
            ops=["torch.add", "torch.sub", "torch.mul", "torch.div"],
            feature="op_capture",
        )
        x = torch.tensor(
            [1, 2, 3], dtype=torch.float16, device="cuda", requires_grad=True
        )
        y = torch.tensor(
            [4, 5, 6], dtype=torch.float16, device="cuda", requires_grad=True
        )

        _test_function(x, y)

    def test_measure_op_time(self):
        op_tools.apply_feature(
            ops=["torch.add", "torch.sub", "torch.mul", "torch.div"],
            feature="measure_op_time",
        )
        x = torch.tensor(
            [1, 2, 3], device="cuda", dtype=torch.float16, requires_grad=True
        )
        y = torch.tensor(
            [4, 5, 6], device="cuda", dtype=torch.float16, requires_grad=True
        )
        _test_function(x, y)

    def test_cast_dtype(self):
        op_tools.apply_feature(
            ops=["torch.add", "torch.sub", "torch.mul", "torch.div"],
            feature="cast_dtype",
        )
        x = torch.randn(4, 5, dtype=torch.float16, device="cuda", requires_grad=True)
        y = torch.rand(4, 5, dtype=torch.float16, device="cuda", requires_grad=True)
        _test_function(x, y)

    def test_condition_fallback(self):
        def condition_func(a, b, **kwargs):
            if a.dtype == torch.float16:
                print(f"fallback beacuse input dtype is {a.dtype}")
                return True
            else:
                print(f"not fallback beacuse input dtype is {a.dtype}")
                return False

        op_tools.apply_feature(
            ops=["torch.add", "torch.sub", "torch.mul", "torch.div"],
            feature="dump_op_args",
            condition_func=condition_func,
        )
        x = torch.tensor(
            [1, 2, 3], dtype=torch.float16, device="cuda", requires_grad=True
        )
        y = torch.tensor(
            [4, 5, 6], dtype=torch.float16, device="cuda", requires_grad=True
        )
        _test_function(x, y)

        x = torch.tensor(
            [1, 2, 3], dtype=torch.float32, device="cuda", requires_grad=True
        )
        y = torch.tensor(
            [4, 5, 6], dtype=torch.float32, device="cuda", requires_grad=True
        )
        _test_function(x, y)

    def test_condition_autocompare(self):
        def condition_func1(a, b, **kwargs):
            if a.dtype == torch.float16:
                print(f"autocompare beacuse input dtype is {a.dtype}")
                return True
            else:
                print(f"not autocompare beacuse input dtype is {a.dtype}")
                return False

        def condition_func2(a, b, **kwargs):
            if a.dim() == 2:
                print(f"autocompare beacuse input dim is {a.dim()}")
                return True
            else:
                print(f"not autocompare beacuse input dim is {a.dim()}")
                return False

        op_tools.apply_feature(
            "torch.add", feature="autocompare", condition_func=condition_func1
        )
        op_tools.apply_feature(
            "torch.sub", feature="autocompare", condition_func=condition_func2
        )
        op_tools.apply_feature(
            "torch.mul", feature="autocompare", condition_func=condition_func1
        )
        op_tools.apply_feature(
            "torch.div", feature="autocompare", condition_func=condition_func2
        )

        x = torch.randn(3, 4, dtype=torch.float16, device="cuda", requires_grad=True)
        y = torch.randn(3, 4, dtype=torch.float16, device="cuda", requires_grad=True)
        _test_function(x, y)

        x = torch.randn(3, 4, dtype=torch.float32, device="cuda", requires_grad=True)
        y = torch.randn(3, 4, dtype=torch.float32, device="cuda", requires_grad=True)
        _test_function(x, y)


if __name__ == "__main__":
    unittest.main()
