import torch
import ditorch
import op_tools
import os


def custom_condition(a, b):
    if a.dtype == torch.float16:
        print("hook enable because a.dtype is float16")
        return True
    elif a.dim() == 2:
        print("hook enable because a.dim() is 2")
        return True
    else:
        print("hook disable")
        return False


x = torch.randn(2, 3, 4, dtype=torch.float16).cuda()
y = torch.randn(4, 2, dtype=torch.float).cuda()
z = torch.randn(2, 3, 4, dtype=torch.float).cuda()

op_tools.apply_feature("torch.add", feature="fallback", condition_func=custom_condition)
torch.add(x, x)

op_tools.apply_feature(
    "torch.sub", feature="autocompare", condition_func=custom_condition
)
torch.sub(y, y)
torch.sub(z, z)

op_tools.apply_feature(
    "torch.mul", feature="op_capture", condition_func=custom_condition
)
torch.mul(x, x)

op_tools.apply_feature(
    "torch.div", feature="cast_dtype", condition_func=custom_condition
)
os.environ["OP_DTYPE_CAST_DICT"] = "torch.float32->torch.float16"
torch.div(y, y)
