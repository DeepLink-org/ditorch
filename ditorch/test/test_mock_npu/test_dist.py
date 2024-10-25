import os
import pytest
import torch
import ditorch
import torch.distributed as dist
import torch.multiprocessing as mp
from functools import partial

# 分布式环境的初始化
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# 清理函数
def cleanup():
    dist.destroy_process_group()


# 分布式进程启动器
def run_distributed_test(test_func, world_size):
    mp.spawn(test_func, args=(world_size,), nprocs=world_size, join=True)

# 测试 all_reduce 的函数
def all_reduce_test(rank, world_size):
    setup(rank, world_size)

    tensor = torch.ones(10).float().cuda(rank) * rank
    print(f"Rank {rank} before all_reduce: {tensor}")

    dist.all_reduce(tensor, op=dist.ReduceOp.AVG)

    print(f"Rank {rank} after all_reduce: {tensor}")
    # 每个进程的张量应该都是相同的，即 0+1 = 1 的和
    expected_tensor = torch.ones(10).float().cuda(rank) * ((sum(range(world_size))) / world_size)
    print(rank, "expected_tensor:", expected_tensor)
    print(rank, "tensor:", tensor)
    assert torch.equal(tensor, expected_tensor), f"Rank {rank} all_reduce failed!"
    cleanup()

# 测试 reduce_scatter 的函数
def reduce_scatter_test(rank, world_size):
    setup(rank, world_size)

    # if world_size is 2,
    # rank 0: [tensor([0]) tensor([1])]
    # rank 1: [tensor([2]) tensor([3])]
    input_tensors = list(torch.arange(world_size * rank, world_size + world_size * rank).float().cuda(rank).chunk(world_size))
    output_tensor = torch.zeros(1).float().cuda(rank)
    print(f"Rank {rank} before reduce_scatter: {input_tensors}")

    dist.reduce_scatter(output_tensor, input_tensors, op=dist.ReduceOp.AVG)
    # expected_tensor:
    # rank 0: tensor([1])
    # rank 1: tensor([2])
    # expected_tensor = torch.tensor((sum((range(world_size)) * world_size) + world_size * rank) / world_size).cuda(rank)
    expected_tensor = torch.tensor([(sum([i * world_size for i in range(world_size)]) + world_size * rank) / world_size]).cuda(rank)
    print(f"Rank {rank} after reduce_scatter: {output_tensor}")
    assert torch.equal(output_tensor, expected_tensor), f"Rank {rank} reduce_scatter failed!"
    cleanup()

def _reduce_scatter_tensor_test(rank, world_size, func):
    setup(rank, world_size)

    # if world_size is 2,
    # rank 0: tensor([0, 1])
    # rank 1: tensor([2, 3])
    input_tensor = torch.arange(world_size * rank, world_size + world_size * rank).float().cuda(rank)
    output_tensor = torch.zeros(int(input_tensor.numel() / world_size)).float().cuda(rank)
    print(f"Rank {rank} before reduce_scatter: {input_tensor}")

    func(output_tensor, input_tensor, op=dist.ReduceOp.AVG)
    # expected_tensor:
    # rank 0: tensor([1])
    # rank 1: tensor([2])
    # expected_tensor = torch.tensor((sum((range(world_size)) * world_size) + world_size * rank) / world_size).cuda(rank)
    expected_tensor = torch.tensor([(sum([i * world_size for i in range(world_size)]) + world_size * rank) / world_size]).cuda(rank)
    print(f"Rank {rank} after reduce_scatter: {output_tensor}")
    assert torch.equal(output_tensor, expected_tensor), f"Rank {rank} reduce_scatter failed!"
    cleanup()

def reduce_scatter_tensor_test(rank, world_size):
    _reduce_scatter_tensor_test(rank, world_size, dist.reduce_scatter_tensor)

def reduce_scatter_base_test(rank, world_size):
    _reduce_scatter_tensor_test(rank, world_size, dist._reduce_scatter_base)

# 测试 reduce 的函数
def reduce_test(rank, world_size):
    setup(rank, world_size)

    tensor = torch.ones(10).cuda(rank) * (rank + 1)
    print(f"Rank {rank} before reduce: {tensor}")

    dist.reduce(tensor, dst=0, op=dist.ReduceOp.AVG)

    if rank == 0:
        expected_tensor = torch.ones(10).cuda(rank) * sum(range(1, world_size + 1)) / world_size
        print(f"Rank {rank} after reduce (on root): {tensor}")
        assert torch.equal(tensor, expected_tensor), "Reduce failed on root!"
    else:
        print(f"Rank {rank} after reduce (non-root): {tensor}")
        # 非 root 进程的结果保持不变
        expected_tensor = torch.ones(10).cuda(rank) * (rank + 1)
        print(f"Rank {rank} after reduce (no root) expected_tensor: {expected_tensor}")
        assert torch.equal(tensor, expected_tensor), "Reduce failed on non-root!"
    cleanup()





# pytest test cases

def test_all_reduce(world_size=2):
    """pytest wrapper for all_reduce test"""
    run_distributed_test(all_reduce_test, world_size)

def test_reduce_scatter(world_size=2):
    """pytest wrapper for reduce_scatter test"""
    run_distributed_test(reduce_scatter_test, world_size)

def test_reduce_scatter_tensor(world_size=2):
    """pytest wrapper for reduce_scatter test"""
    run_distributed_test(reduce_scatter_tensor_test, world_size)

def test__reduce_scatter_base(world_size=2):
    """pytest wrapper for reduce_scatter test"""
    run_distributed_test(reduce_scatter_tensor_test, world_size)

@pytest.mark.parametrize("world_size", [2])
def test_reduce(world_size):
    """pytest wrapper for reduce test"""
    run_distributed_test(reduce_test, world_size)

