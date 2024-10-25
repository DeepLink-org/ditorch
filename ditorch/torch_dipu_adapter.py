# Copyright (c) 2024, DeepLink.
import torch
import torch_dipu
import torch.nn.functional as F
import torch.distributed as dist


def copy_inp(tensor_dest, tensor_src):
    assert tensor_dest.shape == tensor_src.shape, "tensor_dest and tensor_src should have the same shape"
    assert isinstance(tensor_dest, torch.Tensor), "tensor_dest should be a torch.Tensor"
    assert isinstance(tensor_src, torch.Tensor), "tensor_src should be a torch.Tensor"
    if isinstance(tensor_dest, torch.nn.Parameter):
        if isinstance(tensor_src, torch.nn.Parameter):
            tensor_dest.data.copy_(tensor_src.data)
        else:
            tensor_dest.data.copy_(tensor_src)
    else:
        tensor_dest.copy_(tensor_src)


def mock_dist():
    dist_all_reduce = dist.all_reduce
    dist_reduce = dist.reduce
    dist__reduce_scatter_base = dist._reduce_scatter_base

    def dist_reduce_npu(tensor,op=dist.ReduceOp.SUM, group=None, async_op=False, reduce_func = dist_reduce):
        if (op == dist.ReduceOp.AVG):
            handle = reduce_func(tensor, op=dist.ReduceOp.SUM, group=group, async_op=async_op)
            if handle is not None:
                handle.wait()
            world_size = group.size()
            tensor_tmp = tensor / world_size
            copy_inp(tensor, tensor_tmp)
        else:
            handle = reduce_func(tensor, op=op, group=group, async_op=async_op)
        return handle

    def dist_reduce_scatter_npu(output, input ,op=dist.ReduceOp.SUM, group=None, async_op=False):
        if (op == dist.ReduceOp.AVG):
            handle = dist__reduce_scatter_base(output, input, op=dist.ReduceOp.SUM, group=group, async_op=async_op)
            if handle is not None:
                handle.wait()
            world_size = group.size()
            output_tmp = output / world_size
            copy_inp(output, output_tmp)
        else:
            handle = dist__reduce_scatter_base(output, input, op=op, group=group, async_op=async_op)
        return handle

    """================follow must be patch on npu =============="""
    from functools import partial
    dist.all_reduce = partial(dist_reduce_npu, reduce_func= dist_all_reduce)
    dist.reduce = partial(dist_reduce_npu, reduce_func= dist_reduce)
    dist._reduce_scatter_base = dist_reduce_scatter_npu
    """================must be patch on npu end =============="""


mock_dist()
