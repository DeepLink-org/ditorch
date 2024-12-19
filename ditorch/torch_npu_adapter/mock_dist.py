# Copyright (c) 2024, DeepLink.
import torch.nn.functional as F  # noqa
import torch.distributed as dist
from ditorch.utils import is_to_fp32_tensor, copy_inp


def mock_dist(use_fp32=False):
    dist_all_reduce = dist.all_reduce
    dist_reduce = dist.reduce
    dist__reduce_scatter_base = dist._reduce_scatter_base
    dist_reduce_scatter_tensor = dist.reduce_scatter_tensor
    dist_reduce_scatter = dist.reduce_scatter

    @is_to_fp32_tensor(use_fp32)
    def dist_reduce_npu(tensor, dst, op=dist.ReduceOp.SUM, group=None, async_op=False):
        if op == dist.ReduceOp.AVG:
            handle = dist_reduce(tensor, dst, op=dist.ReduceOp.SUM, group=group, async_op=async_op)
            if handle is not None:
                handle.wait()
            if dst == dist.get_rank(group):
                world_size = dist.get_world_size(group)
                tensor_tmp = tensor / world_size
                copy_inp(tensor, tensor_tmp)
        else:
            handle = dist_reduce(tensor, op=op, group=group, async_op=async_op)
        return handle

    @is_to_fp32_tensor(use_fp32)
    def dist_all_reduce_npu(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        if op == dist.ReduceOp.AVG:
            handle = dist_all_reduce(tensor, op=dist.ReduceOp.SUM, group=group, async_op=async_op)
            if handle is not None:
                handle.wait()
            world_size = dist.get_world_size(group)
            tensor_tmp = tensor / world_size
            copy_inp(tensor, tensor_tmp)
        else:
            handle = dist_all_reduce(tensor, op=op, group=group, async_op=async_op)
        return handle

    @is_to_fp32_tensor(use_fp32)
    def dist__reduce_scatter_base_npu(dist_reduce_scatter_func, output, input, op=dist.ReduceOp.SUM, group=None, async_op=False):
        if op == dist.ReduceOp.AVG:
            handle = dist_reduce_scatter_func(output, input, op=dist.ReduceOp.SUM, group=group, async_op=async_op)
            if handle is not None:
                handle.wait()
            world_size = dist.get_world_size(group)
            output_tmp = output / world_size
            copy_inp(output, output_tmp)
        else:
            handle = dist__reduce_scatter_base(output, input, op=op, group=group, async_op=async_op)
        return handle

    """================the following dist op must be mocked on npu =============="""
    from functools import partial

    dist.all_reduce = dist_all_reduce_npu
    dist.reduce = dist_reduce_npu
    dist._reduce_scatter_base = partial(dist__reduce_scatter_base_npu, dist__reduce_scatter_base)
    dist.reduce_scatter = partial(dist__reduce_scatter_base_npu, dist_reduce_scatter)
    dist.reduce_scatter_tensor = partial(dist__reduce_scatter_base_npu, dist_reduce_scatter_tensor)
    """================dist op mocked on npu end ================================="""
