# Copyright (c) 2024, DeepLink.
import torch
import ditorch


def code_to_profile():
    a = torch.rand(10, 20, requires_grad=True).cuda()
    b = a * 2
    c = b + a
    d = c - a
    e = d / c
    sorted, indices = e.sort()  # return torch.return_type.sort
    y = sorted[2:8:2, ::3]
    y.sum().backward()


with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        # torch.profiler.ProfilerActivity.CUDA,
    ]
) as p:
    code_to_profile()


# print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
