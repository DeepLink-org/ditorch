import torch
import ditorch
import unittest


device = torch.device("cuda")


class TestRegisterHook(unittest.TestCase):
    def test_register_hook_on_linear(self):
        func = torch.nn.functional.linear
        input = torch.randn(20, 30, requires_grad=True).to(device)
        weight = torch.randn(40, 30, requires_grad=True, device=device)
        bias = torch.randn(40, requires_grad=True, device=device)
        assert not input.is_leaf
        assert weight.is_leaf
        assert bias.is_leaf
        output = func(input, weight, bias)
        grad_output = None

        def pre_hook(grad_outputs):
            print(f"pre hook:{len(grad_outputs)}")
            nonlocal grad_output
            grad_output = grad_outputs

        input.grad_fn.register_prehook(pre_hook)

        def post_hook(grad_inputs, grad_outputs):
            print(f"post hook:{len(grad_inputs)}")
            print(f"post hook:{len(grad_outputs)}")
            pass

        input.grad_fn.register_hook(post_hook)
        # weight.grad_fn.register_hook(post_hook) # grad_fn is None beacuse weight is a leaf
        assert weight.grad_fn is None

        weitht_grad = None
        bias_grad = None
        input_grad = None

        def weitht_tensor_hook(grad):
            print("weight grad")
            nonlocal weitht_grad
            weitht_grad = grad

        def bias_tensor_hook(grad):
            print("bias grad")
            nonlocal bias_grad
            bias_grad = grad

        def input_tensor_hook(grad):
            print("input grad")
            nonlocal input_grad
            input_grad = grad

        input.register_hook(input_tensor_hook)
        weight.register_hook(weitht_tensor_hook)
        bias.register_hook(bias_tensor_hook)

        output.backward(torch.ones_like(output))
        assert grad_output is not None
        assert weitht_grad is not None
        assert bias_grad is not None
        assert input_grad is not None
        assert torch.allclose(grad_output[0], input_grad)
        assert torch.allclose(weight.grad, weitht_grad)
        assert torch.allclose(bias.grad, bias_grad)
        # assert torch.allclose(input.grad, input_grad) # input.grad is None

    def test_reister_tensor_hook_on_inplace_op(self):
        x = torch.randn(20, 30, requires_grad=True, device=device)
        y = torch.randn(20, 30, requires_grad=True, device=device)
        a = torch.add(x, y) * 2
        b = torch.sub(a, y) / 3
        c = torch.mul(b, a) + 1
        d = torch.div(c, b) - 2

        def get_hook_with_label(label):
            def input_tensor_hook(grad):
                print(f"{label} got grad")

            return input_tensor_hook

        x.register_hook(get_hook_with_label("x"))
        y.register_hook(get_hook_with_label("y"))
        a.register_hook(get_hook_with_label("a"))
        b.register_hook(get_hook_with_label("b"))
        c.register_hook(get_hook_with_label("c"))
        d.register_hook(get_hook_with_label("d"))

        d.backward(torch.ones_like(d))


if __name__ == "__main__":
    unittest.main()
