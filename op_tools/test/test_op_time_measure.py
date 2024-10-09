# Copyright (c) 2024, DeepLink.
# 用了两种使用方式，以及使用了环境变量来规定计算哪些操作的时间
# 应该缺少很多常见操作，也没有模型训练方面的操作
import torch
import ditorch
import op_tools
import os


def f():
    a = torch.rand(10, 20, requires_grad=True).cuda()
    b = a * 2
    c = b + a
    d = c - a
    e = d / c
    sorted, indices = e.sort()  # return torch.return_type.sort
    y = sorted[2:8:2, ::3]
    y.sum().backward()


# usage1
with op_tools.OpTimeMeasure():
    f()

# usage2
capture = op_tools.OpTimeMeasure()
capture.start()
for i in range(3):
    f()
capture.stop()


# usage3
os.environ["OP_TIME_MEASURE_DISABLE_LIST"] = "torch.Tensor.add,torch.Tensor.sub"
capture.start()
f()
capture.stop()


# usage4
os.environ["OP_TIME_MEASURE_DISABLE_LIST"] = ""
os.environ["OP_TIME_MEASURE_LIST"] = "torch.Tensor.sort"  # only capture these op
capture.start()
f()
capture.stop()


# usage5
os.environ["OP_TIME_MEASURE_DISABLE_LIST"] = ""
os.environ["OP_TIME_MEASURE_LIST"] = ""  # 空
capture.start()
f()
capture.stop()

# usage6
os.environ["OP_TIME_MEASURE_DISABLE_LIST"] = "torch.Tensor.sort"
os.environ["OP_TIME_MEASURE_LIST"] = "torch.Tensor.sort,torch.Tensor.add"  # 重叠
capture.start()
f()
capture.stop()

# usage7
os.environ["OP_TIME_MEASURE_DISABLE_LIST"] = "torch.Tensor.add,torch.Tensor.sub"
if "OP_TIME_MEASURE_LIST" in os.environ:
    del os.environ["OP_TIME_MEASURE_LIST"]  # 删除
capture.start()
f()
capture.stop()
