# Copyright (c) 2024, DeepLink.
framework = None
try:
    from ditorch import torch_npu_adapter

    framework = "torch_npu:" + torch_npu_adapter.torch_npu.__version__
except:
    pass
try:
    from ditorch import torch_dipu_adapter

    framework = "torch_dipu:" + torch_dipu_adapter.torch_dipu.__version__

except:
    pass

try:
    from ditorch import torch_mlu_adapter

    framework = "torch_mlu:" + torch_mlu_adapter.torch_mlu.__version__
except:
    pass

print(f"ditorch.framework: {framework}")
