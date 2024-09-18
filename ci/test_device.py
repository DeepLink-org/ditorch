import torch
import ditorch

if torch.npu.is_available():
    a = torch.tensor([1, 2, 3], device='cuda')
    b = torch.tensor([1, 2, 4], device='cuda')
    res = torch.add(a, b)
    print("npu is available")
    print("res = ", res)
else:
    print("oooops, something wrong!")