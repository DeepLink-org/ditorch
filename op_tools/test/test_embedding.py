from contextlib import AbstractContextManager
from typing import Any
import ditorch
import torch
import torch.nn as nn
import unittest
import op_tools


class TestEmbedding(unittest.TestCase):

    def test_embedding1(self):
        with op_tools.OpCapture():
            n, d, m = 3, 5, 7
            embedding = nn.Embedding(n, d, max_norm=True, device="cuda")
            W = torch.randn((m, d), requires_grad=True, device="cuda")
            idx = torch.tensor([1, 2]).cuda()
            a = (
                embedding.weight.clone() @ W.t()
            )  # weight must be cloned for this to be differentiable
            b = embedding(idx) @ W.t()  # modifies weight in-place
            out = a.unsqueeze(0) + b.unsqueeze(1)
            loss = out.sigmoid().prod()
            loss.backward()

    def test_embedding2(self):
        with op_tools.OpAutoCompare():
            # an Embedding module containing 10 tensors of size 3
            embedding = nn.Embedding(10, 3)
            # a batch of 2 samples of 4 indices each
            input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
            output = embedding(input)

    def test_embedding3(self):
        with op_tools.OpAutoCompare():
            # example with padding_idx
            embedding = nn.Embedding(10, 3, padding_idx=0)
            input = torch.LongTensor([[0, 2, 0, 5]])
            output = embedding(input)

    def test_embedding4(self):
        with op_tools.OpAutoCompare():
            # example of changing `pad` vector
            padding_idx = 0
            embedding = nn.Embedding(3, 3, padding_idx=padding_idx)
            with torch.no_grad():
                embedding.weight[padding_idx] = torch.ones(3)


if __name__ == "__main__":
    unittest.main()
