# Copyright (c) 2024, DeepLink.
import torch_mlu  # noqa: F401
import torch


def mock_jiterator():
    class jiterator:
        pass

    def _create_jit_fn(code_string: str, **kwargs):
        def do_nothing(*args, **kwargs):
            print("do nothing, not support _create_jit_fn on camb")

        return do_nothing

    def _create_multi_output_jit_fn(code_string: str, num_outputs: int, **kwargs):
        def do_nothing(*args, **kwargs):
            print("do nothing, not support _create_multi_output_jit_fn on camb")

        return do_nothing

    torch.cuda.jiterator = jiterator
    torch.cuda.jiterator._create_multi_output_jit_fn = _create_multi_output_jit_fn
    torch.cuda.jiterator._create_jit_fn = _create_jit_fn


def mock_has_magma():
    torch.cuda.has_magma = False


def mock_stream():
    torch._C._cuda_setStream = torch_mlu._MLUC._mlu_setStream
    torch._C._cuda_setDevice = torch_mlu._MLUC._mlu_setDevice


def mock_runtime():
    mock_jiterator()
    mock_has_magma()
    mock_stream()
