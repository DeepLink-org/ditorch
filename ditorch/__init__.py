try:
    from ditorch import torch_npu_adapter

    print("ditorch use torch_npu")
except:
    pass

try:
    from ditorch import torch_dipu

    print("ditorch use torch_dipu")
except:
    pass
