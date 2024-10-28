import torch
import torch.nn.functional as F  # noqa
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
    dist_reduce_scatter_tensor = dist.reduce_scatter_tensor
    dist_reduce_scatter = dist.reduce_scatter

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

    """================follow must be patch on npu =============="""
    from functools import partial

    dist.all_reduce = dist_all_reduce_npu
    dist.reduce = dist_reduce_npu
    dist._reduce_scatter_base = partial(dist__reduce_scatter_base_npu, dist__reduce_scatter_base)
    dist.reduce_scatter = partial(dist__reduce_scatter_base_npu, dist_reduce_scatter)
    dist.reduce_scatter_tensor = partial(dist__reduce_scatter_base_npu, dist_reduce_scatter_tensor)
    """================must be patch on npu end =============="""


def mock():
    mock_dist()
