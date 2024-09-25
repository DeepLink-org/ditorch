# Copyright (c) 2024, DeepLink.
import torch_mlu  # noqa: F401
from torch_mlu.utils.model_transfer import transfer  # noqa: F401
import torch


class jiterator:
    pass


def _create_jit_fn(code_string: str, **kwargs):
    def do_nothing(*args, **kwargs):
        print("do nothing, not support _create_jit_fn on camb")

    return do_nothing


torch.cuda.jiterator = jiterator
torch.cuda.jiterator._create_jit_fn = _create_jit_fn


def _create_multi_output_jit_fn(code_string: str, num_outputs: int, **kwargs):
    def do_nothing(*args, **kwargs):
        print("do nothing, not support _create_jit_fn on camb")

    return do_nothing


torch.cuda.jiterator._create_multi_output_jit_fn = _create_multi_output_jit_fn


torch.cuda.has_magma = False
