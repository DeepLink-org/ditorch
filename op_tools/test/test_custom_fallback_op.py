# Copyright (c) 2024, DeepLink.
import torch
import op_tools
import unittest


class TestCustomFallbackOp(unittest.TestCase):

    def test_fallback_op(self):
        op_tools.apply_feature(
            ops=["torch.add", "torch.sub", "torch.mul", "torch.div"], feature="fallback"
        )
        x = torch.tensor([1, 2, 3], device="cuda")
        y = torch.tensor([4, 5, 6], device="cuda")
        z = torch.add(x, y)
        assert z.is_cpu == False

        n = torch.sub(x, y)
        assert n.is_cpu == False

        m = torch.mul(x, y)
        assert m.is_cpu == False

        p = x + y
        assert p.is_cpu == False

    def test_dump_all_args(self):
        op_tools.apply_feature(
            ops=["torch.add", "torch.sub", "torch.mul", "torch.div"],
            feature="autocompare",
        )
        x = torch.tensor([1, 2, 3], device="cuda")
        y = torch.tensor([4, 5, 6], device="cuda")

        z = torch.add(x, y)
        z = torch.sub(x, y)
        z = torch.mul(x, y)
        z = torch.div(x, y)

    def test_condition_fallback(self):
        def condition_func(a, b, **kwargs):
            if a.dtype == torch.float16:
                print(f"fallback beacuse input dtype is float16")
                return True
            else:
                print(f"not fallback beacuse input dtype is {a.dtype}")
                return False

        op_tools.apply_feature(
            ops=["torch.add", "torch.sub", "torch.mul", "torch.div"],
            feature="dump_op_args",
            condition_func=condition_func,
        )
        x = torch.tensor([1, 2, 3], device="cuda").half()
        y = torch.tensor([4, 5, 6], device="cuda").half()
        z = torch.add(x, y)
        z = torch.sub(x, y)
        z = torch.mul(x, y)
        z = torch.div(x, y)

        x = torch.tensor([1, 2, 3], device="cuda").float()
        y = torch.tensor([4, 5, 6], device="cuda").float()
        z = torch.add(x, y)
        z = torch.sub(x, y)
        z = torch.mul(x, y)
        z = torch.div(x, y)

    def test_condition_autocompare(self):
        def condition_func1(a, b, **kwargs):
            if a.dtype == torch.float16:
                print(f"autocompare beacuse input dtype is float16")
                return True
            else:
                print(f"not autocompare beacuse input dtype is {a.dtype}")
                return False

        def condition_func2(a, b, **kwargs):
            if a.dim() == 2:
                print(f"autocompare beacuse input dim is 2")
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

        x = torch.randn(3, 4, device="cuda").half()
        y = torch.randn(3, 4, device="cuda").half()

        z = torch.add(x, y)
        z = torch.sub(x, y)
        z = torch.mul(x, y)
        z = torch.div(x, y)


if __name__ == "__main__":
    unittest.main()
