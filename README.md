# 设备无关torch, 旨在屏蔽各硬件厂商torch差异，为用户提供一致使用体验
![](ditorch.png)

## 核心点
### 1. 两行代码即可像官方pytorch一样在国产芯片上使用pytorch
```
import torch
import ditorch
```


### 2. 提供模型训练过程中需要的基础工具，解决模型训练过程中出现的痛点问题
[算子工具](op_tools/README.md)

#### 算子参数抓取工具
抓取模型真实训练过程中真实的输入输出

```
# usage1
import op_tools
capture = op_tools.OpCapture()
capture.start()
code_snippet_to_capture
capture.stop()
...
```

```
# usage2
import op_tools
with op_tools.OpCapture():
    code_snippet_to_capture()
```

##### 抓取前向和反向的所有输入输出

```
op_capture_result/0/2024-08-06--11-41/torch.Tensor.to/8/input.pth saved
op_capture_result/0/2024-08-06--11-41/torch.Tensor.to/8/output.pth saved
apply OpCaptureHook on torch.Tensor.mul
op_capture_result/0/2024-08-06--11-41/torch.Tensor.mul/9/input.pth saved
op_capture_result/0/2024-08-06--11-41/torch.Tensor.mul/9/output.pth saved
apply OpCaptureHook on torch.Tensor.add
op_capture_result/0/2024-08-06--11-41/torch.Tensor.add/10/input.pth saved
op_capture_result/0/2024-08-06--11-41/torch.Tensor.add/10/output.pth saved
apply OpCaptureHook on torch.Tensor.sub
op_capture_result/0/2024-08-06--11-41/torch.Tensor.sub/11/input.pth saved
op_capture_result/0/2024-08-06--11-41/torch.Tensor.sub/11/output.pth saved
apply OpCaptureHook on torch.Tensor.div
op_capture_result/0/2024-08-06--11-41/torch.Tensor.div/12/input.pth saved
op_capture_result/0/2024-08-06--11-41/torch.Tensor.div/12/output.pth saved
apply OpCaptureHook on torch.Tensor.sort
op_capture_result/0/2024-08-06--11-41/torch.Tensor.sort/13/input.pth saved
op_capture_result/0/2024-08-06--11-41/torch.Tensor.sort/13/output.pth saved
apply OpCaptureHook on torch.Tensor.sum
op_capture_result/0/2024-08-06--11-41/torch.Tensor.sum/14/input.pth saved
op_capture_result/0/2024-08-06--11-41/torch.Tensor.sum/14/output.pth saved
skip OpCaptureHook on torch.Tensor.backward
op_capture_result/0/2024-08-06--11-41/torch.Tensor.sum/14/grad_inputs.pth saved
op_capture_result/0/2024-08-06--11-41/torch.Tensor.sum/14/grad_outputs.pth saved
op_capture_result/0/2024-08-06--11-41/torch.Tensor.sort/13/grad_inputs.pth saved
op_capture_result/0/2024-08-06--11-41/torch.Tensor.sort/13/grad_outputs.pth saved
op_capture_result/0/2024-08-06--11-41/torch.Tensor.to/8/grad_inputs.pth saved
op_capture_result/0/2024-08-06--11-41/torch.Tensor.to/8/grad_outputs.pth saved
...
```

##### 只抓取sort算子的参数，忽略其他算子 OP_CAPTURE_LIST=torch.Tensor.sort
```
skip OpCaptureHook on torch.device
skip OpCaptureHook on torch.Tensor.to
skip OpCaptureHook on torch.Tensor.mul
skip OpCaptureHook on torch.Tensor.add
skip OpCaptureHook on torch.Tensor.sub
skip OpCaptureHook on torch.Tensor.div
apply OpCaptureHook on torch.Tensor.sort
op_capture_result/0/2024-08-06--11-41/torch.Tensor.sort/34/input.pth saved
op_capture_result/0/2024-08-06--11-41/torch.Tensor.sort/34/output.pth saved
op_capture_result/0/2024-08-06--11-41/torch.Tensor.sort/34/grad_inputs.pth saved
op_capture_result/0/2024-08-06--11-41/torch.Tensor.sort/34/grad_outputs.pth saved
...
```

##### 排除指定算子，抓取所有其他算子 OP_CAPTURE_DISABLE_LIST="torch.Tensor.add,torch.Tensor.sub"
```
apply OpCaptureHook on torch.Tensor.to
op_capture_result/0/2024-08-06--11-46/torch.Tensor.to/29/input.pth saved
op_capture_result/0/2024-08-06--11-46/torch.Tensor.to/29/output.pth saved
apply OpCaptureHook on torch.Tensor.mul
op_capture_result/0/2024-08-06--11-46/torch.Tensor.mul/30/input.pth saved
op_capture_result/0/2024-08-06--11-46/torch.Tensor.mul/30/output.pth saved
skip OpCaptureHook on torch.Tensor.add
skip OpCaptureHook on torch.Tensor.sub
apply OpCaptureHook on torch.Tensor.div
op_capture_result/0/2024-08-06--11-46/torch.Tensor.div/31/input.pth saved
op_capture_result/0/2024-08-06--11-46/torch.Tensor.div/31/output.pth saved
apply OpCaptureHook on torch.Tensor.sort
op_capture_result/0/2024-08-06--11-46/torch.Tensor.sort/32/input.pth saved
op_capture_result/0/2024-08-06--11-46/torch.Tensor.sort/32/output.pth saved
apply OpCaptureHook on torch.Tensor.sum
op_capture_result/0/2024-08-06--11-46/torch.Tensor.sum/33/input.pth saved
op_capture_result/0/2024-08-06--11-46/torch.Tensor.sum/33/output.pth saved
...
```
#### 精度分析工具
离线分析 + 实时精度对比
1. 用模型训练过程中真实输入输出，离线对比
2. 模型训练时实时与cpu对比分析精度
##### 基于InternEvo + ditorch + torch_npu 在华为910B上实时精度分析输出片段
```
# usage1
import op_tools
with op_tools.OpAutoCompare():
    code_snippet_to_autocompare()
```

```
# usage1
import op_tools
autocompare = op_tools.OpAutoCompare()
autocompare.start()
code_snippet_to_autocompare()
autocompare.stop()
```

```
OpAutoCompareHook: torch.nn.functional.linear                         allclose: False    max_diff:          0.003906250
OpAutoCompareHook: torch.nn.functional.linear                         input: {'args': ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20075956404224}, {'shape': torch.Size([2048, 2048]), 'stride': (2048, 1), 'numel': 4194304, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': True, 'layout': 'torch.strided', 'data': 20078077673472}, 'None')}
OpAutoCompareHook: torch.nn.functional.linear                         output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076023513600},)
op_capture_result/torch.nn.functional.linear/93/device/input.pth saved
op_capture_result/torch.nn.functional.linear/93/device/output.pth saved
op_capture_result/torch.nn.functional.linear/93/cpu/input.pth saved
op_capture_result/torch.nn.functional.linear/93/cpu/output.pth saved
OpAutoCompareHook: torch.Tensor.contiguous                            allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.is_complex                            allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.nn.functional.dropout                        allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.add                                   allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.to                                    allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.to                                    allclose: True    max_diff:          0.000000000
...
OpAutoCompareHook: torch.Tensor.contiguous                            allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.is_complex                            allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.add                                   allclose: True    max_diff:          0.000000000
skip OpAutoCompareHook on None
OpAutoCompareHook: torch.Tensor.float                                 allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.to                                    allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.pow                                   allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.mean                                  allclose: True    max_diff:          0.000000238
OpAutoCompareHook: torch.Tensor.add                                   allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.rsqrt                                        allclose: True    max_diff:          0.000000179
OpAutoCompareHook: torch.Tensor.mul                                   allclose: True    max_diff:          0.000000000
skip OpAutoCompareHook on torch.Tensor.dtype.__get__
skip OpAutoCompareHook on torch.Tensor.dtype.__get__
OpAutoCompareHook: torch.Tensor.to                                    allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.mul                                   allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.nn.functional.normalize                      allclose: True    max_diff:          0.000488281
skip OpAutoCompareHook on torch.Tensor.dtype.__get__
skip OpAutoCompareHook on torch.Tensor.requires_grad.__get__
OpAutoCompareHook: torch.Tensor.contiguous                            allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.contiguous                            allclose: True    max_diff:          0.000000000
...
OpAutoCompareHook: torch.Tensor.float                                 allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.contiguous                            allclose: True    max_diff:          0.000000000
skip OpAutoCompareHook on torch.Tensor.size
OpAutoCompareHook: torch.Tensor.view                                  allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.contiguous                            allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.view                                  allclose: True    max_diff:          0.000000000
skip OpAutoCompareHook on torch.Tensor.dtype.__get__
OpAutoCompareHook: torch.nn.functional.cross_entropy                  allclose: True    max_diff:          0.000003815
OpAutoCompareHook: torch.Tensor.ne                                    allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.sum                                   allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.sum                                   allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.div                                   allclose: True    max_diff:          0.000000000
skip OpAutoCompareHook on None
OpAutoCompareHook: torch.Tensor.detach                                allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.sum                                   allclose: True    max_diff:          0.015625000
OpAutoCompareHook: torch.Tensor.add_                                  allclose: True    max_diff:          0.000000000
skip OpAutoCompareHook on torch.Tensor.numel
OpAutoCompareHook: torch.Tensor.add_                                  allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.detach                                allclose: True    max_diff:          0.000000000
skip OpAutoCompareHook on None
skip OpAutoCompareHook on torch.device
skip OpAutoCompareHook on None
OpAutoCompareHook: torch.Tensor.div_                                  allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.div_                                  allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.add_                                  allclose: True    max_diff:          0.000000000
skip OpAutoCompareHook on None
skip OpAutoCompareHook on torch.Tensor.size
OpAutoCompareHook: torch.Tensor.view                                  allclose: True    max_diff:          0.000000000
skip OpAutoCompareHook on torch.Tensor.size
OpAutoCompareHook: torch.Tensor.view                                  allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.view                                  allclose: True    max_diff:          0.000000000
skip OpAutoCompareHook on torch.Tensor.shape.__get__
OpAutoCompareHook: torch.max                                      0th allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.max                                      1th allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.max                                      0th allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.max                                      1th allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.eq                                    allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.argmax                                allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.add                                   allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.eq                                    allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.logical_and                                  allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.long                                  allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.ne                                    allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.long                                  allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.dim                                   allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.dim                                   allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.dim                                   allclose: True    max_diff:          0.000000000
skip OpAutoCompareHook on torch.Tensor.size
OpAutoCompareHook: torch.Tensor.expand                                allclose: True    max_diff:          0.000000000
skip OpAutoCompareHook on torch.Tensor.size
skip OpAutoCompareHook on torch.Tensor.numel
OpAutoCompareHook: torch.Tensor.max                                   allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.int                                   allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.add                                   allclose: True    max_diff:          0.000000000
skip OpAutoCompareHook on torch.Tensor.dtype.__get__
skip OpAutoCompareHook on torch.Tensor.device.__get__
OpAutoCompareHook: torch.Tensor.__index__                             allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.__index__                             allclose: True    max_diff:          0.000000000
skip OpAutoCompareHook on None
OpAutoCompareHook: torch.Tensor.scatter_add_                          allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.dim                                   allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.dim                                   allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.dim                                   allclose: True    max_diff:          0.000000000
skip OpAutoCompareHook on torch.Tensor.size
OpAutoCompareHook: torch.Tensor.expand                                allclose: True    max_diff:          0.000000000
skip OpAutoCompareHook on torch.Tensor.size
skip OpAutoCompareHook on torch.Tensor.numel
OpAutoCompareHook: torch.Tensor.max                                   allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.int                                   allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.add                                   allclose: True    max_diff:          0.000000000
...
OpAutoCompareHook: torch.functional.norm                              allclose: True    max_diff:          0.001953125
OpAutoCompareHook: torch.functional.norm                              allclose: True    max_diff:         71.062500000
OpAutoCompareHook: torch.functional.norm                              allclose: True    max_diff:        237.750000000
OpAutoCompareHook: torch.functional.norm                              allclose: True    max_diff:          0.000488281
OpAutoCompareHook: torch.functional.norm                              allclose: False    max_diff:       1473.750000000
OpAutoCompareHook: torch.functional.norm                              input: {'args': ({'shape': torch.Size([2048, 8192]), 'stride': (8192, 1), 'numel': 16777216, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067372762112},), 'kwargs': {'p': '2', 'dim': 'None', 'keepdim': 'False', 'out': 'None', 'dtype': 'None'}}
OpAutoCompareHook: torch.functional.norm                              output: ({'shape': torch.Size([]), 'stride': (), 'numel': 1, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067180820992},)
op_capture_result/torch.functional.norm/93/device/input.pth saved
op_capture_result/torch.functional.norm/93/device/output.pth saved
op_capture_result/torch.functional.norm/93/cpu/input.pth saved
op_capture_result/torch.functional.norm/93/cpu/output.pth saved
...
OpAutoCompareHook: torch.triu                                         allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.bool                                  allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.tolist                            0th allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.tolist                            1th allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.tolist                            2th allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.tolist                            3th allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.tolist                            0th allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.tolist                            1th allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.tolist                            2th allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.tolist                            3th allclose: True    max_diff:          0.000000000
skip OpAutoCompareHook on npu.npu_fusion_attention
...
OpAutoCompareHook: torch.Tensor.div                                   allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.item                                  allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.div                                   allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.exp                                          allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.item                                  allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.float                                 allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.float                                 allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.add                                   allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.div                                   allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.item                                  allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.item                                  allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.float                                 allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.float                                 allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.add                                   allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.div                                   allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.item                                  allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.item                                  allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.float                                 allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.float                                 allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.add                                   allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.div                                   allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.item                                  allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.item                                  allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.fill_                                 allclose: True    max_diff:          0.000000000
...
OpAutoCompareHook: torch.Tensor.to                                    allclose: False    max_diff:          0.000000000 Inconsistent dtypes: torch.float32 torch.float64
OpAutoCompareHook: torch.Tensor.mean                                  allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.to                                    allclose: False    max_diff:          0.000000000 Inconsistent dtypes: torch.float32 torch.float64
OpAutoCompareHook: torch.Tensor.std                                   allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.to                                    allclose: False    max_diff:          0.000000000 Inconsistent dtypes: torch.float32 torch.float64
OpAutoCompareHook: torch.Tensor.mean                                  allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.to                                    allclose: False    max_diff:          0.000000000 Inconsistent dtypes: torch.float32 torch.float64
OpAutoCompareHook: torch.Tensor.std                                   allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.to                                    allclose: False    max_diff:          0.000000000 Inconsistent dtypes: torch.float32 torch.float64
OpAutoCompareHook: torch.Tensor.mean                                  allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.to                                    allclose: False    max_diff:          0.000000000 Inconsistent dtypes: torch.float32 torch.float64
OpAutoCompareHook: torch.Tensor.std                                   allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.to                                    allclose: False    max_diff:          0.000000000 Inconsistent dtypes: torch.float32 torch.float64
OpAutoCompareHook: torch.Tensor.mean                                  allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.to                                    allclose: False    max_diff:          0.000000000 Inconsistent dtypes: torch.float32 torch.float64
```

#### 性能分析工具
离线分析 + 实时分析
用模型训练过程中真实的输入输出分析算子和通信的耗时，分析出性能瓶颈
```
# 测量算子耗时（输入为使用算子抓取工具在模型训练时抓取到的真实数据）
 python run_op_from_data.py /op_capture_result/torch.Tensor.div/2278281/5  --run_times 10
ditorch.framework: torch_npu:2.1.0.post3
/op_capture_result/torch.Tensor.div/2278281/5
SyncExecuteTimer: torch.Tensor.div forward elasped 72.62969017 ms
SyncExecuteTimer: torch.Tensor.div backward elasped 171.01812363 ms
SyncExecuteTimer: torch.Tensor.div forward elasped 0.08916855 ms
SyncExecuteTimer: torch.Tensor.div backward elasped 2.09069252 ms
SyncExecuteTimer: torch.Tensor.div forward elasped 0.06723404 ms
SyncExecuteTimer: torch.Tensor.div backward elasped 3.06391716 ms
SyncExecuteTimer: torch.Tensor.div forward elasped 0.05483627 ms
SyncExecuteTimer: torch.Tensor.div backward elasped 1.41191483 ms
SyncExecuteTimer: torch.Tensor.div forward elasped 0.05912781 ms
SyncExecuteTimer: torch.Tensor.div backward elasped 1.36375427 ms
SyncExecuteTimer: torch.Tensor.div forward elasped 0.05030632 ms
SyncExecuteTimer: torch.Tensor.div backward elasped 1.45721436 ms
SyncExecuteTimer: torch.Tensor.div forward elasped 0.04959106 ms
SyncExecuteTimer: torch.Tensor.div backward elasped 1.62410736 ms
SyncExecuteTimer: torch.Tensor.div forward elasped 0.05149841 ms
SyncExecuteTimer: torch.Tensor.div backward elasped 1.35779381 ms
SyncExecuteTimer: torch.Tensor.div forward elasped 0.04506111 ms
SyncExecuteTimer: torch.Tensor.div backward elasped 1.29985809 ms
SyncExecuteTimer: torch.Tensor.div forward elasped 0.04172325 ms
SyncExecuteTimer: torch.Tensor.div backward elasped 1.24096870 ms
```

##### 只跑指定算子3遍前向
```
ditorch/op_tools# python run_op_from_data.py /op_capture_result/torch.Tensor.div/2278281/5  --run_times 3 --only_run_forward True
ditorch.framework: torch_npu:2.1.0.post3
/deeplink_afs/zhaoguochun/ditorch/op_tools/op_capture_result/torch.Tensor.div/2278281/5
SyncExecuteTimer: torch.Tensor.div forward elasped 91.06540680 ms
SyncExecuteTimer: torch.Tensor.div forward elasped 0.24318695 ms
SyncExecuteTimer: torch.Tensor.div forward elasped 0.07224083 ms
```

##### 模型训练时算子耗时分析 (前向 + 反向)
```
# usage1
import op_tools
with op_tools.OpTimeMeasure():
    code_snippet_to_time_measure()
```

```
# usage2
import op_tools
timemeasure = op_tools.OpTimeMeasure()
timemeasure.start()
code_snippet_to_time_measure()
timemeasure.end()
```

```
...
OpTimeMeasureHook: torch.Tensor.is_floating_point forward elasped:  0.00929832 ms     input: {'args': ({'shape': torch.Size([8192, 2048]), 'stride': (2048, 1), 'numel': 16777216, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': True, 'layout': 'torch.strided', 'data': 20067618127872},)} output: {'args': ('True',)}
OpTimeMeasureHook: torch.Tensor.to                forward elasped:  0.01168251 ms     input: {'args': ({'shape': torch.Size([8192, 2048]), 'stride': (2048, 1), 'numel': 16777216, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': True, 'layout': 'torch.strided', 'data': 20067618127872}, 'None', 'torch.bfloat16', 'False')} output: {'args': ({'shape': torch.Size([8192, 2048]), 'stride': (2048, 1), 'numel': 16777216, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': True, 'layout': 'torch.strided', 'data': 20067618127872},)}
...
OpTimeMeasureHook: torch.Tensor.is_complex        forward elasped:  0.00929832 ms     input: {'args': ({'shape': torch.Size([]), 'stride': (), 'numel': 1, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067179829760},)} output: {'args': ('False',)}
OpTimeMeasureHook: torch.Tensor.item              forward elasped:  0.02098083 ms     input: {'args': ({'shape': torch.Size([]), 'stride': (), 'numel': 1, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067179829760},)} output: {'args': ('0.16592097282409668',)}
skip OpTimeMeasureHook on None
OpTimeMeasureHook: torch.Tensor.to                forward elasped:  0.03743172 ms     input: {'args': ({'shape': torch.Size([]), 'stride': (), 'numel': 1, 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 33419042112}, 'npu:0')} output: {'args': ({'shape': torch.Size([]), 'stride': (), 'numel': 1, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067179829760},)}
OpTimeMeasureHook: torch.Tensor.is_complex        forward elasped:  0.00929832 ms     input: {'args': ({'shape': torch.Size([]), 'stride': (), 'numel': 1, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067179829760},)} output: {'args': ('False',)}
OpTimeMeasureHook: torch.Tensor.item              forward elasped:  0.01811981 ms     input: {'args': ({'shape': torch.Size([]), 'stride': (), 'numel': 1, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067179829760},)} output: {'args': ('4.982948303222656e-05',)}
OpTimeMeasureHook: torch.Tensor.to                forward elasped:  0.02336502 ms     input: {'args': ({'shape': torch.Size([]), 'stride': (), 'numel': 1, 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 33419044160}, 'npu:0')} output: {'args': ({'shape': torch.Size([]), 'stride': (), 'numel': 1, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067179829760},)}
OpTimeMeasureHook: torch.Tensor.is_complex        forward elasped:  0.00810623 ms     input: {'args': ({'shape': torch.Size([]), 'stride': (), 'numel': 1, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067179829760},)} output: {'args': ('False',)}
OpTimeMeasureHook: torch.Tensor.item              forward elasped:  0.01740456 ms     input: {'args': ({'shape': torch.Size([]), 'stride': (), 'numel': 1, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067179829760},)} output: {'args': ('0.004927396774291992',)}
2024-08-07 20:40:04,803 INFO record_metrics.py:373 in record_execution_times -- {'import_time': 0.07344746589660645, 'init_comm_time': 12.286690711975098, 'init_model_time': 0.8780200481414795, 'load_data_time': 36.91646957397461, 'init_optim_time': 0.16592097282409668, 'load_ckpt_time': 4.982948303222656e-05, 'init_trainer_time': 0.004927396774291992}
OpTimeMeasureHook: torch.Tensor.random_           forward elasped:  0.05078316 ms     input: {'args': ({'shape': torch.Size([]), 'stride': (), 'numel': 1, 'dtype': 'torch.int64', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 33419039680},), 'kwargs': {'generator': 'None'}} output: {'args': ({'shape': torch.Size([]), 'stride': (), 'numel': 1, 'dtype': 'torch.int64', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 33419039680},)}
...
OpTimeMeasureHook: torch.nn.init.normal_          forward elasped:  701.74193382 ms     input: {'args': (), 'kwargs': {'tensor': {'shape': torch.Size([92544, 2048]), 'stride': (2048, 1), 'numel': 189530112, 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': True, 'layout': 'torch.strided', 'data': 140665192181824}, 'mean': '0.0', 'std': '0.02'}} output: {'args': ({'shape': torch.Size([92544, 2048]), 'stride': (2048, 1), 'numel': 189530112, 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': True, 'layout': 'torch.strided', 'data': 140665192181824},)}
...

```

#### 算子fallback 能力
```
# usage 1
with op_tools.OpFallback():
    code_snippet_op_to_be_fallbacked()
```

```
# usage 2
fallback = op_tools.OpFallback()
fallback.start()
code_snippet_op_to_be_fallbacked()
fallback.end()
```

##### 只fallback 指定算子 export OP_FALLBACK_LIST="torch.nn.functional.linear"
```
skip OpFallbackHook on torch.Tensor.float
skip OpFallbackHook on torch.Tensor.add
skip OpFallbackHook on torch.Tensor.div
skip OpFallbackHook on torch.Tensor.item
skip OpFallbackHook on torch.Tensor.item
skip OpFallbackHook on torch.Tensor.float
skip OpFallbackHook on torch.Tensor.float
skip OpFallbackHook on torch.Tensor.add
skip OpFallbackHook on torch.Tensor.div
skip OpFallbackHook on torch.Tensor.item
skip OpFallbackHook on torch.Tensor.item
skip OpFallbackHook on torch.Tensor.fill_
skip OpFallbackHook on torch.Tensor.fill_
skip OpFallbackHook on torch.Tensor.fill_
skip OpFallbackHook on torch.Tensor.fill_
skip OpFallbackHook on torch.Tensor.fill_
skip OpFallbackHook on torch.Tensor.fill_
skip OpFallbackHook on torch.Tensor.div
skip OpFallbackHook on torch.Tensor.item
skip OpFallbackHook on torch.Tensor.item
skip OpFallbackHook on torch.Tensor.div
skip OpFallbackHook on torch.Tensor.item
skip OpFallbackHook on torch.Tensor.div
skip OpFallbackHook on torch.Tensor.item
skip OpFallbackHook on torch.Tensor.div
skip OpFallbackHook on torch.Tensor.item
skip OpFallbackHook on torch.Tensor.fill_
skip OpFallbackHook on torch.Tensor.fill_
skip OpFallbackHook on torch.Tensor.fill_
skip OpFallbackHook on torch.Tensor.fill_
skip OpFallbackHook on torch.Tensor.is_complex
skip OpFallbackHook on torch.Tensor.is_complex
skip OpFallbackHook on torch.Tensor.fill_
skip OpFallbackHook on torch.Tensor.fill_
skip OpFallbackHook on torch.Tensor.item
skip OpFallbackHook on torch.Tensor.numel
skip OpFallbackHook on torch.Tensor.dim
skip OpFallbackHook on torch.Tensor.unbind
skip OpFallbackHook on torch.Tensor.__len__
skip OpFallbackHook on torch.Tensor.dim
skip OpFallbackHook on torch.Tensor.unbind
skip OpFallbackHook on torch.Tensor.sub
skip OpFallbackHook on torch.Tensor.max
skip OpFallbackHook on torch.Tensor.item
skip OpFallbackHook on torch.Tensor.dim
skip OpFallbackHook on torch.Tensor.unbind
skip OpFallbackHook on torch.Tensor.__len__
skip OpFallbackHook on torch.Tensor.dim
skip OpFallbackHook on torch.Tensor.unbind
skip OpFallbackHook on torch.Tensor.__len__
skip OpFallbackHook on torch.Tensor.__len__
skip OpFallbackHook on torch.Tensor.detach
skip OpFallbackHook on torch.Tensor.cpu
skip OpFallbackHook on torch.Tensor.numpy
skip OpFallbackHook on torch.Tensor.detach
skip OpFallbackHook on torch.Tensor.cpu
skip OpFallbackHook on torch.Tensor.numpy
skip OpFallbackHook on torch.Tensor.detach
skip OpFallbackHook on torch.Tensor.cpu
skip OpFallbackHook on torch.Tensor.numpy
skip OpFallbackHook on torch.Tensor.detach
skip OpFallbackHook on torch.Tensor.cpu
skip OpFallbackHook on torch.Tensor.numpy
...
OpFallbackHook: torch.nn.functional.linear                         input: {'args': ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076203868160}, {'shape': torch.Size([4096, 2048]), 'stride': (2048, 1), 'numel': 8388608, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': True, 'layout': 'torch.strided', 'data': 20077985398784}, 'None')}
OpFallbackHook: torch.nn.functional.linear                         output: ({'shape': torch.Size([1, 16384, 4096]), 'stride': (67108864, 4096, 1), 'numel': 67108864, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20075820089344},) cpu output: ({'shape': torch.Size([1, 16384, 4096]), 'stride': (67108864, 4096, 1), 'numel': 67108864, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139743270527040},) dtype_convert_back_dict:{}
skip OpFallbackHook on torch.Tensor.shape.__get__
...
```

##### fallback指定算子以外所有算子（export OP_FALLBACK_DISABLE_LIST="torch.nn.functional.linear"）
```
...
skip OpFallbackHook on torch.nn.functional.linear
OpFallbackHook: torch.Tensor.float                                 input: {'args': ({'shape': torch.Size([1, 16384, 92544]), 'stride': (1516240896, 92544, 1), 'numel': 1516240896, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': True, 'layout': 'torch.strided', 'data': 20081119592448},)}
OpFallbackHook: torch.Tensor.float                                 output: ({'shape': torch.Size([1, 16384, 92544]), 'stride': (1516240896, 92544, 1), 'numel': 1516240896, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': True, 'layout': 'torch.strided', 'data': 20100446945280},) cpu output: ({'shape': torch.Size([1, 16384, 92544]), 'stride': (1516240896, 92544, 1), 'numel': 1516240896, 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': True, 'layout': 'torch.strided', 'data': 140152888873024},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.contiguous                            input: {'args': ({'shape': torch.Size([1, 16384, 92544]), 'stride': (1516240896, 92544, 1), 'numel': 1516240896, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': True, 'layout': 'torch.strided', 'data': 20100446945280},)}
OpFallbackHook: torch.Tensor.contiguous                            output: ({'shape': torch.Size([1, 16384, 92544]), 'stride': (1516240896, 92544, 1), 'numel': 1516240896, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': True, 'layout': 'torch.strided', 'data': 20106889396224},) cpu output: ({'shape': torch.Size([1, 16384, 92544]), 'stride': (1516240896, 92544, 1), 'numel': 1516240896, 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': True, 'layout': 'torch.strided', 'data': 140155921358912},) dtype_convert_back_dict:{}
skip OpFallbackHook on torch.Tensor.size
OpFallbackHook: torch.Tensor.view                                  input: {'args': ({'shape': torch.Size([1, 16384, 92544]), 'stride': (1516240896, 92544, 1), 'numel': 1516240896, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': True, 'layout': 'torch.strided', 'data': 20106889396224}, '-1', '92544')}
OpFallbackHook: torch.Tensor.view                                  output: ({'shape': torch.Size([16384, 92544]), 'stride': (92544, 1), 'numel': 1516240896, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': True, 'layout': 'torch.strided', 'data': 20113331847168},) cpu output: ({'shape': torch.Size([16384, 92544]), 'stride': (92544, 1), 'numel': 1516240896, 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': True, 'layout': 'torch.strided', 'data': 140155921358912},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.contiguous                            input: {'args': ({'shape': torch.Size([1, 16384]), 'stride': (16384, 1), 'numel': 16384, 'dtype': 'torch.int64', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067180535808},)}
OpFallbackHook: torch.Tensor.contiguous                            output: ({'shape': torch.Size([1, 16384]), 'stride': (16384, 1), 'numel': 16384, 'dtype': 'torch.int64', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067179877888},) cpu output: ({'shape': torch.Size([1, 16384]), 'stride': (16384, 1), 'numel': 16384, 'dtype': 'torch.int64', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 33663304832},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.view                                  input: {'args': ({'shape': torch.Size([1, 16384]), 'stride': (16384, 1), 'numel': 16384, 'dtype': 'torch.int64', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067179877888}, '-1')}
...

```

##### fallback所有算子时部分输出
```
OpFallbackHook: torch.nn.functional.linear                         input: {'args': ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074851205120}, {'shape': torch.Size([2048, 2048]), 'stride': (2048, 1), 'numel': 4194304, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067599254528}, 'None')}
OpFallbackHook: torch.nn.functional.linear                         output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074920411136},) cpu output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739386380352},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.contiguous                            input: {'args': ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074920411136},)}
OpFallbackHook: torch.Tensor.contiguous                            output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074851205120},) cpu output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739506286656},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.is_complex                            input: {'args': ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074851205120},)}
OpFallbackHook: torch.Tensor.is_complex                            output: ('False',) cpu output: ('False',) dtype_convert_back_dict:{}
OpFallbackHook: torch.nn.functional.dropout                        input: {'args': ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074851205120},), 'kwargs': {'p': '0', 'training': 'True', 'inplace': 'False'}}
OpFallbackHook: torch.nn.functional.dropout                        output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067756539904},) cpu output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739506286656},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.add                                   input: {'args': ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067756539904}, {'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076199673856})}
OpFallbackHook: torch.Tensor.add                                   output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074712793088},) cpu output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739319267392},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.to                                    input: {'args': ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074712793088}, 'torch.float32')}
OpFallbackHook: torch.Tensor.to                                    output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076338085888},) cpu output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739319271488},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.to                                    input: {'args': ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076338085888}, 'torch.float32')}
OpFallbackHook: torch.Tensor.to                                    output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076474400768},) cpu output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739319271488},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.pow                                   input: {'args': ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076474400768}, '2')}
OpFallbackHook: torch.Tensor.pow                                   output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076610715648},) cpu output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139730855391296},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.mean                                  input: {'args': ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076610715648}, ('-1',)), 'kwargs': {'keepdim': 'True'}}
OpFallbackHook: torch.Tensor.mean                                  output: ({'shape': torch.Size([1, 16384, 1]), 'stride': (16384, 1, 1), 'numel': 16384, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067180141056},) cpu output: ({'shape': torch.Size([1, 16384, 1]), 'stride': (16384, 1, 1), 'numel': 16384, 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 33561021952},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.add                                   input: {'args': ({'shape': torch.Size([1, 16384, 1]), 'stride': (16384, 1, 1), 'numel': 16384, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067180141056}, '1e-05')}
OpFallbackHook: torch.Tensor.add                                   output: ({'shape': torch.Size([1, 16384, 1]), 'stride': (16384, 1, 1), 'numel': 16384, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067181455872},) cpu output: ({'shape': torch.Size([1, 16384, 1]), 'stride': (16384, 1, 1), 'numel': 16384, 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 33561087616},) dtype_convert_back_dict:{}
OpFallbackHook: torch.rsqrt                                        input: {'args': ({'shape': torch.Size([1, 16384, 1]), 'stride': (16384, 1, 1), 'numel': 16384, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067181455872},)}
OpFallbackHook: torch.rsqrt                                        output: ({'shape': torch.Size([1, 16384, 1]), 'stride': (16384, 1, 1), 'numel': 16384, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067181521920},) cpu output: ({'shape': torch.Size([1, 16384, 1]), 'stride': (16384, 1, 1), 'numel': 16384, 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 33561218944},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.mul                                   input: {'args': ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076338085888}, {'shape': torch.Size([1, 16384, 1]), 'stride': (16384, 1, 1), 'numel': 16384, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067181521920})}
OpFallbackHook: torch.Tensor.mul                                   output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076474400768},) cpu output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139730855391296},) dtype_convert_back_dict:{}
skip OpFallbackHook on torch.Tensor.dtype.__get__
skip OpFallbackHook on torch.Tensor.dtype.__get__
OpFallbackHook: torch.Tensor.to                                    input: {'args': ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076474400768}, 'torch.bfloat16')}
OpFallbackHook: torch.Tensor.to                                    output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074781999104},) cpu output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739506286656},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.mul                                   input: {'args': ({'shape': torch.Size([2048]), 'stride': (1,), 'numel': 2048, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': True, 'layout': 'torch.strided', 'data': 20067179833856}, {'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074781999104})}
OpFallbackHook: torch.Tensor.mul                                   output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074920411136},) cpu output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739386380352},) dtype_convert_back_dict:{}
skip OpFallbackHook on torch.Tensor.dtype.__get__
skip OpFallbackHook on torch.Tensor.requires_grad.__get__
OpFallbackHook: torch.Tensor.contiguous                            input: {'args': ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074920411136},)}
OpFallbackHook: torch.Tensor.contiguous                            output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067756539904},) cpu output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739506286656},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.contiguous                            input: {'args': ({'shape': torch.Size([8192, 2048]), 'stride': (2048, 1), 'numel': 16777216, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': True, 'layout': 'torch.strided', 'data': 20067236446208},)}
OpFallbackHook: torch.Tensor.contiguous                            output: ({'shape': torch.Size([8192, 2048]), 'stride': (2048, 1), 'numel': 16777216, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067720888832},) cpu output: ({'shape': torch.Size([8192, 2048]), 'stride': (2048, 1), 'numel': 16777216, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 33821921472},) dtype_convert_back_dict:{}
skip OpFallbackHook on torch.Tensor.shape.__get__
skip OpFallbackHook on torch.Tensor.shape.__get__
skip OpFallbackHook on torch.Tensor.shape.__get__
OpFallbackHook: torch.nn.functional.linear                         input: {'args': ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067756539904}, {'shape': torch.Size([8192, 2048]), 'stride': (2048, 1), 'numel': 16777216, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067720888832}, 'None')}
OpFallbackHook: torch.nn.functional.linear                         output: ({'shape': torch.Size([1, 16384, 8192]), 'stride': (134217728, 8192, 1), 'numel': 134217728, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20075132223488},) cpu output: ({'shape': torch.Size([1, 16384, 8192]), 'stride': (134217728, 8192, 1), 'numel': 134217728, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139730721173568},) dtype_convert_back_dict:{}
skip OpFallbackHook on torch.Tensor.dtype.__get__
skip OpFallbackHook on torch.Tensor.requires_grad.__get__
OpFallbackHook: torch.Tensor.contiguous                            input: {'args': ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074920411136},)}
OpFallbackHook: torch.Tensor.contiguous                            output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067756539904},) cpu output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739506286656},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.contiguous                            input: {'args': ({'shape': torch.Size([8192, 2048]), 'stride': (2048, 1), 'numel': 16777216, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': True, 'layout': 'torch.strided', 'data': 20067305652736},)}
OpFallbackHook: torch.Tensor.contiguous                            output: ({'shape': torch.Size([8192, 2048]), 'stride': (2048, 1), 'numel': 16777216, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067720888832},) cpu output: ({'shape': torch.Size([8192, 2048]), 'stride': (2048, 1), 'numel': 16777216, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 33821921472},) dtype_convert_back_dict:{}
skip OpFallbackHook on torch.Tensor.shape.__get__
skip OpFallbackHook on torch.Tensor.shape.__get__
skip OpFallbackHook on torch.Tensor.shape.__get__
OpFallbackHook: torch.nn.functional.linear                         input: {'args': ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067756539904}, {'shape': torch.Size([8192, 2048]), 'stride': (2048, 1), 'numel': 16777216, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067720888832}, 'None')}
OpFallbackHook: torch.nn.functional.linear                         output: ({'shape': torch.Size([1, 16384, 8192]), 'stride': (134217728, 8192, 1), 'numel': 134217728, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20075402756096},) cpu output: ({'shape': torch.Size([1, 16384, 8192]), 'stride': (134217728, 8192, 1), 'numel': 134217728, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139730721173568},) dtype_convert_back_dict:{}
OpFallbackHook: torch.nn.functional.silu                           input: {'args': ({'shape': torch.Size([1, 16384, 8192]), 'stride': (134217728, 8192, 1), 'numel': 134217728, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20075132223488},), 'kwargs': {'inplace': 'False'}}
OpFallbackHook: torch.nn.functional.silu                           output: ({'shape': torch.Size([1, 16384, 8192]), 'stride': (134217728, 8192, 1), 'numel': 134217728, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076824625152},) cpu output: ({'shape': torch.Size([1, 16384, 8192]), 'stride': (134217728, 8192, 1), 'numel': 134217728, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139730452734016},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.mul                                   input: {'args': ({'shape': torch.Size([1, 16384, 8192]), 'stride': (134217728, 8192, 1), 'numel': 134217728, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076824625152}, {'shape': torch.Size([1, 16384, 8192]), 'stride': (134217728, 8192, 1), 'numel': 134217728, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20075402756096})}
OpFallbackHook: torch.Tensor.mul                                   output: ({'shape': torch.Size([1, 16384, 8192]), 'stride': (134217728, 8192, 1), 'numel': 134217728, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20077095157760},) cpu output: ({'shape': torch.Size([1, 16384, 8192]), 'stride': (134217728, 8192, 1), 'numel': 134217728, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139730184294464},) dtype_convert_back_dict:{}
skip OpFallbackHook on torch.Tensor.dtype.__get__
skip OpFallbackHook on torch.Tensor.requires_grad.__get__
OpFallbackHook: torch.Tensor.contiguous                            input: {'args': ({'shape': torch.Size([1, 16384, 8192]), 'stride': (134217728, 8192, 1), 'numel': 134217728, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20077095157760},)}
OpFallbackHook: torch.Tensor.contiguous                            output: ({'shape': torch.Size([1, 16384, 8192]), 'stride': (134217728, 8192, 1), 'numel': 134217728, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076824625152},) cpu output: ({'shape': torch.Size([1, 16384, 8192]), 'stride': (134217728, 8192, 1), 'numel': 134217728, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139730721173568},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.contiguous                            input: {'args': ({'shape': torch.Size([2048, 8192]), 'stride': (8192, 1), 'numel': 16777216, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': True, 'layout': 'torch.strided', 'data': 20067272097792},)}
OpFallbackHook: torch.Tensor.contiguous                            output: ({'shape': torch.Size([2048, 8192]), 'stride': (8192, 1), 'numel': 16777216, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067720888832},) cpu output: ({'shape': torch.Size([2048, 8192]), 'stride': (8192, 1), 'numel': 16777216, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 33821921472},) dtype_convert_back_dict:{}
skip OpFallbackHook on torch.Tensor.shape.__get__
skip OpFallbackHook on torch.Tensor.shape.__get__
skip OpFallbackHook on torch.Tensor.shape.__get__
OpFallbackHook: torch.nn.functional.linear                         input: {'args': ({'shape': torch.Size([1, 16384, 8192]), 'stride': (134217728, 8192, 1), 'numel': 134217728, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076824625152}, {'shape': torch.Size([2048, 8192]), 'stride': (8192, 1), 'numel': 16777216, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067720888832}, 'None')}
OpFallbackHook: torch.nn.functional.linear                         output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067756539904},) cpu output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739506286656},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.contiguous                            input: {'args': ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067756539904},)}
OpFallbackHook: torch.Tensor.contiguous                            output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074781999104},) cpu output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739506286656},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.is_complex                            input: {'args': ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074781999104},)}
OpFallbackHook: torch.Tensor.is_complex                            output: ('False',) cpu output: ('False',) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.add                                   input: {'args': ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074781999104}, {'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074712793088})}
OpFallbackHook: torch.Tensor.add                                   output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067756539904},) cpu output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739319267392},) dtype_convert_back_dict:{}
skip OpFallbackHook on None
skip OpFallbackHook on torch.Tensor.requires_grad.__get__
skip OpFallbackHook on torch.device
skip OpFallbackHook on torch.device
skip OpFallbackHook on torch.Tensor.clone
skip OpFallbackHook on torch.Tensor.clone
skip OpFallbackHook on torch.Tensor.clone
skip OpFallbackHook on torch.Tensor.clone
skip OpFallbackHook on torch.Tensor.clone
skip OpFallbackHook on None
OpFallbackHook: torch.nn.functional.dropout                        input: {'args': ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': True, 'layout': 'torch.strided', 'data': 20067756539904},), 'kwargs': {'p': '0', 'training': 'True', 'inplace': 'False'}}
OpFallbackHook: torch.nn.functional.dropout                        output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074712793088},) cpu output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739506286656},) dtype_convert_back_dict:{}
skip OpFallbackHook on torch.Tensor.dtype.__get__
OpFallbackHook: torch.Tensor.to                                    input: {'args': ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074712793088},), 'kwargs': {'dtype': 'torch.bfloat16'}}
OpFallbackHook: torch.Tensor.to                                    output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074781999104},) cpu output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739506286656},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.to                                    input: {'args': ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074781999104}, 'torch.float32')}
OpFallbackHook: torch.Tensor.to                                    output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076338085888},) cpu output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739319271488},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.pow                                   input: {'args': ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076338085888}, '2')}
OpFallbackHook: torch.Tensor.pow                                   output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076474400768},) cpu output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139730855391296},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.mean                                  input: {'args': ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076474400768}, ('-1',)), 'kwargs': {'keepdim': 'True'}}
OpFallbackHook: torch.Tensor.mean                                  output: ({'shape': torch.Size([1, 16384, 1]), 'stride': (16384, 1, 1), 'numel': 16384, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067180141056},) cpu output: ({'shape': torch.Size([1, 16384, 1]), 'stride': (16384, 1, 1), 'numel': 16384, 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 33561284608},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.add                                   input: {'args': ({'shape': torch.Size([1, 16384, 1]), 'stride': (16384, 1, 1), 'numel': 16384, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067180141056}, '1e-05')}
OpFallbackHook: torch.Tensor.add                                   output: ({'shape': torch.Size([1, 16384, 1]), 'stride': (16384, 1, 1), 'numel': 16384, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067181455872},) cpu output: ({'shape': torch.Size([1, 16384, 1]), 'stride': (16384, 1, 1), 'numel': 16384, 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 33561415936},) dtype_convert_back_dict:{}
OpFallbackHook: torch.rsqrt                                        input: {'args': ({'shape': torch.Size([1, 16384, 1]), 'stride': (16384, 1, 1), 'numel': 16384, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067181455872},)}
OpFallbackHook: torch.rsqrt                                        output: ({'shape': torch.Size([1, 16384, 1]), 'stride': (16384, 1, 1), 'numel': 16384, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067181521920},) cpu output: ({'shape': torch.Size([1, 16384, 1]), 'stride': (16384, 1, 1), 'numel': 16384, 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 33557410880},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.mul                                   input: {'args': ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074781999104}, {'shape': torch.Size([1, 16384, 1]), 'stride': (16384, 1, 1), 'numel': 16384, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067181521920})}
OpFallbackHook: torch.Tensor.mul                                   output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076338085888},) cpu output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139730855391296},) dtype_convert_back_dict:{}
skip OpFallbackHook on torch.Tensor.dtype.__get__
skip OpFallbackHook on torch.Tensor.dtype.__get__
OpFallbackHook: torch.Tensor.to                                    input: {'args': ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076338085888}, 'torch.bfloat16')}
OpFallbackHook: torch.Tensor.to                                    output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074851205120},) cpu output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739506286656},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.mul                                   input: {'args': ({'shape': torch.Size([2048]), 'stride': (1,), 'numel': 2048, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': True, 'layout': 'torch.strided', 'data': 20067179838464}, {'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074851205120})}
OpFallbackHook: torch.Tensor.mul                                   output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074920411136},) cpu output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739386380352},) dtype_convert_back_dict:{}
skip OpFallbackHook on torch.Tensor.dtype.__get__
skip OpFallbackHook on torch.Tensor.requires_grad.__get__
OpFallbackHook: torch.Tensor.contiguous                            input: {'args': ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074920411136},)}
OpFallbackHook: torch.Tensor.contiguous                            output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074781999104},) cpu output: ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739506286656},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.contiguous                            input: {'args': ({'shape': torch.Size([4096, 2048]), 'stride': (2048, 1), 'numel': 8388608, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': True, 'layout': 'torch.strided', 'data': 20067209183744},)}
OpFallbackHook: torch.Tensor.contiguous                            output: ({'shape': torch.Size([4096, 2048]), 'stride': (2048, 1), 'numel': 8388608, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067599254528},) cpu output: ({'shape': torch.Size([4096, 2048]), 'stride': (2048, 1), 'numel': 8388608, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 33771589440},) dtype_convert_back_dict:{}
skip OpFallbackHook on torch.Tensor.shape.__get__
skip OpFallbackHook on torch.Tensor.shape.__get__
skip OpFallbackHook on torch.Tensor.shape.__get__
OpFallbackHook: torch.nn.functional.linear                         input: {'args': ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074781999104}, {'shape': torch.Size([4096, 2048]), 'stride': (2048, 1), 'numel': 8388608, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067599254528}, 'None')}
OpFallbackHook: torch.nn.functional.linear                         output: ({'shape': torch.Size([1, 16384, 4096]), 'stride': (67108864, 4096, 1), 'numel': 67108864, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076338085888},) cpu output: ({'shape': torch.Size([1, 16384, 4096]), 'stride': (67108864, 4096, 1), 'numel': 67108864, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739319271488},) dtype_convert_back_dict:{}
skip OpFallbackHook on torch.Tensor.shape.__get__
skip OpFallbackHook on torch.Tensor.shape.__get__
OpFallbackHook: torch.Tensor.reshape                               input: {'args': ({'shape': torch.Size([1, 16384, 4096]), 'stride': (67108864, 4096, 1), 'numel': 67108864, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076338085888}, ['1', '16384', '8', '4', '128'])}
OpFallbackHook: torch.Tensor.reshape                               output: ({'shape': torch.Size([1, 16384, 8, 4, 128]), 'stride': (67108864, 4096, 512, 128, 1), 'numel': 67108864, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076474400768},) cpu output: ({'shape': torch.Size([1, 16384, 8, 4, 128]), 'stride': (67108864, 4096, 512, 128, 1), 'numel': 67108864, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739319271488},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.__getitem__                           input: {'args': ({'shape': torch.Size([1, 16384, 8, 4, 128]), 'stride': (67108864, 4096, 512, 128, 1), 'numel': 67108864, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076474400768}, ('Ellipsis', 'slice(None, 2, None)', 'slice(None, None, None)'))}
OpFallbackHook: torch.Tensor.__getitem__                           output: ({'shape': torch.Size([1, 16384, 8, 2, 128]), 'stride': (33554432, 2048, 256, 128, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074781999104},) cpu output: ({'shape': torch.Size([1, 16384, 8, 2, 128]), 'stride': (67108864, 4096, 512, 128, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739319271488},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.__getitem__                           input: {'args': ({'shape': torch.Size([1, 16384, 8, 4, 128]), 'stride': (67108864, 4096, 512, 128, 1), 'numel': 67108864, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076474400768}, ('Ellipsis', '-2', 'slice(None, None, None)'))}
OpFallbackHook: torch.Tensor.__getitem__                           output: ({'shape': torch.Size([1, 16384, 8, 128]), 'stride': (16777216, 1024, 128, 1), 'numel': 16777216, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067720888832},) cpu output: ({'shape': torch.Size([1, 16384, 8, 128]), 'stride': (67108864, 4096, 512, 1), 'numel': 16777216, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739319272000},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.__getitem__                           input: {'args': ({'shape': torch.Size([1, 16384, 8, 4, 128]), 'stride': (67108864, 4096, 512, 128, 1), 'numel': 67108864, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076474400768}, ('Ellipsis', '-1', 'slice(None, None, None)'))}
OpFallbackHook: torch.Tensor.__getitem__                           output: ({'shape': torch.Size([1, 16384, 8, 128]), 'stride': (16777216, 1024, 128, 1), 'numel': 16777216, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074677141504},) cpu output: ({'shape': torch.Size([1, 16384, 8, 128]), 'stride': (67108864, 4096, 512, 1), 'numel': 16777216, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739319272256},) dtype_convert_back_dict:{}
skip OpFallbackHook on torch.Tensor.shape.__get__
skip OpFallbackHook on torch.Tensor.shape.__get__
OpFallbackHook: torch.Tensor.reshape                               input: {'args': ({'shape': torch.Size([1, 16384, 8, 2, 128]), 'stride': (33554432, 2048, 256, 128, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074781999104}, ['1', '16384', '16', '128'])}
OpFallbackHook: torch.Tensor.reshape                               output: ({'shape': torch.Size([1, 16384, 16, 128]), 'stride': (33554432, 2048, 128, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074851205120},) cpu output: ({'shape': torch.Size([1, 16384, 16, 128]), 'stride': (33554432, 2048, 128, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739506286656},) dtype_convert_back_dict:{}
skip OpFallbackHook on torch.Tensor.device.__get__
skip OpFallbackHook on torch.Tensor.dtype.__get__
skip OpFallbackHook on None
skip OpFallbackHook on torch.Tensor.device.__get__
OpFallbackHook: torch.Tensor.to                                    input: {'args': ({'shape': torch.Size([64]), 'stride': (1,), 'numel': 64, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067179824640},), 'kwargs': {'device': 'npu:0'}}
OpFallbackHook: torch.Tensor.to                                    output: ({'shape': torch.Size([64]), 'stride': (1,), 'numel': 64, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067180145664},) cpu output: ({'shape': torch.Size([64]), 'stride': (1,), 'numel': 64, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067180145664},) dtype_convert_back_dict:{}
OpFallbackHook: torch.outer                                        input: {'args': ({'shape': torch.Size([1025]), 'stride': (1,), 'numel': 1025, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067180141056}, {'shape': torch.Size([64]), 'stride': (1,), 'numel': 64, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067180145664})}
OpFallbackHook: torch.outer                                        output: ({'shape': torch.Size([1025, 64]), 'stride': (64, 1), 'numel': 65600, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067181455872},) cpu output: ({'shape': torch.Size([1025, 64]), 'stride': (64, 1), 'numel': 65600, 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 33557476544},) dtype_convert_back_dict:{}
OpFallbackHook: torch.cos                                          input: {'args': ({'shape': torch.Size([1025, 64]), 'stride': (64, 1), 'numel': 65600, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067181455872},)}
OpFallbackHook: torch.cos                                          output: ({'shape': torch.Size([1025, 64]), 'stride': (64, 1), 'numel': 65600, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067180667392},) cpu output: ({'shape': torch.Size([1025, 64]), 'stride': (64, 1), 'numel': 65600, 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 33558001600},) dtype_convert_back_dict:{}
skip OpFallbackHook on torch.Tensor.dtype.__get__
OpFallbackHook: torch.Tensor.to                                    input: {'args': ({'shape': torch.Size([1025, 64]), 'stride': (64, 1), 'numel': 65600, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067180667392}, 'torch.bfloat16')}
OpFallbackHook: torch.Tensor.to                                    output: ({'shape': torch.Size([1025, 64]), 'stride': (64, 1), 'numel': 65600, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067181718528},) cpu output: ({'shape': torch.Size([1025, 64]), 'stride': (64, 1), 'numel': 65600, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 33561350272},) dtype_convert_back_dict:{}
OpFallbackHook: torch.sin                                          input: {'args': ({'shape': torch.Size([1025, 64]), 'stride': (64, 1), 'numel': 65600, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067181455872},)}
OpFallbackHook: torch.sin                                          output: ({'shape': torch.Size([1025, 64]), 'stride': (64, 1), 'numel': 65600, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067180667392},) cpu output: ({'shape': torch.Size([1025, 64]), 'stride': (64, 1), 'numel': 65600, 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 33558264128},) dtype_convert_back_dict:{}
skip OpFallbackHook on torch.Tensor.dtype.__get__
OpFallbackHook: torch.Tensor.to                                    input: {'args': ({'shape': torch.Size([1025, 64]), 'stride': (64, 1), 'numel': 65600, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067180667392}, 'torch.bfloat16')}
OpFallbackHook: torch.Tensor.to                                    output: ({'shape': torch.Size([1025, 64]), 'stride': (64, 1), 'numel': 65600, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067180930048},) cpu output: ({'shape': torch.Size([1025, 64]), 'stride': (64, 1), 'numel': 65600, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 33558264128},) dtype_convert_back_dict:{}
skip OpFallbackHook on torch.Tensor.shape.__get__
skip OpFallbackHook on torch.Tensor.shape.__get__
skip OpFallbackHook on torch.Tensor.shape.__get__
OpFallbackHook: torch.Tensor.__getitem__                           input: {'args': ({'shape': torch.Size([1, 16384, 16, 128]), 'stride': (33554432, 2048, 128, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074851205120}, ('Ellipsis', 'slice(None, 128, None)'))}
OpFallbackHook: torch.Tensor.__getitem__                           output: ({'shape': torch.Size([1, 16384, 16, 128]), 'stride': (33554432, 2048, 128, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074781999104},) cpu output: ({'shape': torch.Size([1, 16384, 16, 128]), 'stride': (33554432, 2048, 128, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739506286656},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.chunk                                 input: {'args': ({'shape': torch.Size([1, 16384, 16, 128]), 'stride': (33554432, 2048, 128, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074781999104}, '2'), 'kwargs': {'dim': '-1'}}
OpFallbackHook: torch.Tensor.chunk                                 output: (({'shape': torch.Size([1, 16384, 16, 64]), 'stride': (16777216, 1024, 64, 1), 'numel': 16777216, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074989617152}, {'shape': torch.Size([1, 16384, 16, 64]), 'stride': (16777216, 1024, 64, 1), 'numel': 16777216, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20075023172096}),) cpu output: (({'shape': torch.Size([1, 16384, 16, 64]), 'stride': (33554432, 2048, 128, 1), 'numel': 16777216, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739506286656}, {'shape': torch.Size([1, 16384, 16, 64]), 'stride': (33554432, 2048, 128, 1), 'numel': 16777216, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739506286784}),) dtype_convert_back_dict:{}
skip OpFallbackHook on None
OpFallbackHook: torch.Tensor.__getitem__                           input: {'args': ({'shape': torch.Size([1, 16384, 16, 128]), 'stride': (33554432, 2048, 128, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076199673856}, ('Ellipsis', 'slice(None, 128, None)'))}
OpFallbackHook: torch.Tensor.__getitem__                           output: ({'shape': torch.Size([1, 16384, 16, 128]), 'stride': (33554432, 2048, 128, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076268879872},) cpu output: ({'shape': torch.Size([1, 16384, 16, 128]), 'stride': (33554432, 2048, 128, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739506286656},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.chunk                                 input: {'args': ({'shape': torch.Size([1, 16384, 16, 128]), 'stride': (33554432, 2048, 128, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076268879872}, '2'), 'kwargs': {'dim': '-1'}}
OpFallbackHook: torch.Tensor.chunk                                 output: (({'shape': torch.Size([1, 16384, 16, 64]), 'stride': (16777216, 1024, 64, 1), 'numel': 16777216, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076747030528}, {'shape': torch.Size([1, 16384, 16, 64]), 'stride': (16777216, 1024, 64, 1), 'numel': 16777216, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076780585472}),) cpu output: (({'shape': torch.Size([1, 16384, 16, 64]), 'stride': (33554432, 2048, 128, 1), 'numel': 16777216, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739506286656}, {'shape': torch.Size([1, 16384, 16, 64]), 'stride': (33554432, 2048, 128, 1), 'numel': 16777216, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739506286784}),) dtype_convert_back_dict:{}
skip OpFallbackHook on torch.Tensor.shape.__get__
skip OpFallbackHook on torch.Tensor.shape.__get__
OpFallbackHook: torch.Tensor.reshape                               input: {'args': ({'shape': torch.Size([16384, 64]), 'stride': (64, 1), 'numel': 1048576, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067599254528}, ['16384', '1', '64'])}
OpFallbackHook: torch.Tensor.reshape                               output: ({'shape': torch.Size([16384, 1, 64]), 'stride': (64, 64, 1), 'numel': 1048576, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067603449856},) cpu output: ({'shape': torch.Size([16384, 1, 64]), 'stride': (64, 64, 1), 'numel': 1048576, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 33555245696},) dtype_convert_back_dict:{}
skip OpFallbackHook on torch.Tensor.shape.__get__
skip OpFallbackHook on torch.Tensor.shape.__get__
OpFallbackHook: torch.Tensor.reshape                               input: {'args': ({'shape': torch.Size([16384, 64]), 'stride': (64, 1), 'numel': 1048576, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067601352192}, ['16384', '1', '64'])}
OpFallbackHook: torch.Tensor.reshape                               output: ({'shape': torch.Size([16384, 1, 64]), 'stride': (64, 64, 1), 'numel': 1048576, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067605547520},) cpu output: ({'shape': torch.Size([16384, 1, 64]), 'stride': (64, 64, 1), 'numel': 1048576, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 33599621824},) dtype_convert_back_dict:{}
skip OpFallbackHook on torch.Tensor.device.__get__
skip OpFallbackHook on torch.Tensor.device.__get__
skip OpFallbackHook on torch.Tensor.device.__get__
skip OpFallbackHook on torch.Tensor.device.__get__
skip OpFallbackHook on torch.Tensor.dtype.__get__
skip OpFallbackHook on torch.Tensor.dtype.__get__
skip OpFallbackHook on torch.Tensor.dtype.__get__
skip OpFallbackHook on torch.Tensor.dtype.__get__
skip OpFallbackHook on torch.Tensor.size
skip OpFallbackHook on torch.Tensor.size
skip OpFallbackHook on torch.Tensor.size
skip OpFallbackHook on torch.Tensor.size
OpFallbackHook: torch.Tensor.float                                 input: {'args': ({'shape': torch.Size([1, 16384, 16, 64]), 'stride': (16777216, 1024, 64, 1), 'numel': 16777216, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074989617152},)}
OpFallbackHook: torch.Tensor.float                                 output: ({'shape': torch.Size([1, 16384, 16, 64]), 'stride': (16777216, 1024, 64, 1), 'numel': 16777216, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076338085888},) cpu output: ({'shape': torch.Size([1, 16384, 16, 64]), 'stride': (16777216, 1024, 64, 1), 'numel': 16777216, 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739506286656},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.float                                 input: {'args': ({'shape': torch.Size([1, 16384, 16, 64]), 'stride': (16777216, 1024, 64, 1), 'numel': 16777216, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20075023172096},)}
OpFallbackHook: torch.Tensor.float                                 output: ({'shape': torch.Size([1, 16384, 16, 64]), 'stride': (16777216, 1024, 64, 1), 'numel': 16777216, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076405195264},) cpu output: ({'shape': torch.Size([1, 16384, 16, 64]), 'stride': (16777216, 1024, 64, 1), 'numel': 16777216, 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739506286656},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.float                                 input: {'args': ({'shape': torch.Size([16384, 1, 64]), 'stride': (64, 64, 1), 'numel': 1048576, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067603449856},)}
OpFallbackHook: torch.Tensor.float                                 output: ({'shape': torch.Size([16384, 1, 64]), 'stride': (64, 64, 1), 'numel': 1048576, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067607645184},) cpu output: ({'shape': torch.Size([16384, 1, 64]), 'stride': (64, 64, 1), 'numel': 1048576, 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 33788366784},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.float                                 input: {'args': ({'shape': torch.Size([16384, 1, 64]), 'stride': (64, 64, 1), 'numel': 1048576, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067605547520},)}
OpFallbackHook: torch.Tensor.float                                 output: ({'shape': torch.Size([16384, 1, 64]), 'stride': (64, 64, 1), 'numel': 1048576, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067611840000},) cpu output: ({'shape': torch.Size([16384, 1, 64]), 'stride': (64, 64, 1), 'numel': 1048576, 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 33696091456},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.mul                                   input: {'args': ({'shape': torch.Size([1, 16384, 16, 64]), 'stride': (16777216, 1024, 64, 1), 'numel': 16777216, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076338085888}, {'shape': torch.Size([16384, 1, 64]), 'stride': (64, 64, 1), 'numel': 1048576, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067607645184})}
OpFallbackHook: torch.Tensor.mul                                   output: ({'shape': torch.Size([1, 16384, 16, 64]), 'stride': (16777216, 1024, 64, 1), 'numel': 16777216, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076610715648},) cpu output: ({'shape': torch.Size([1, 16384, 16, 64]), 'stride': (16777216, 1024, 64, 1), 'numel': 16777216, 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739386380352},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.mul                                   input: {'args': ({'shape': torch.Size([1, 16384, 16, 64]), 'stride': (16777216, 1024, 64, 1), 'numel': 16777216, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076405195264}, {'shape': torch.Size([16384, 1, 64]), 'stride': (64, 64, 1), 'numel': 1048576, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067611840000})}
OpFallbackHook: torch.Tensor.mul                                   output: ({'shape': torch.Size([1, 16384, 16, 64]), 'stride': (16777216, 1024, 64, 1), 'numel': 16777216, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076677825024},) cpu output: ({'shape': torch.Size([1, 16384, 16, 64]), 'stride': (16777216, 1024, 64, 1), 'numel': 16777216, 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739386380352},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.sub                                   input: {'args': ({'shape': torch.Size([1, 16384, 16, 64]), 'stride': (16777216, 1024, 64, 1), 'numel': 16777216, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076610715648}, {'shape': torch.Size([1, 16384, 16, 64]), 'stride': (16777216, 1024, 64, 1), 'numel': 16777216, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076677825024})}
OpFallbackHook: torch.Tensor.sub                                   output: ({'shape': torch.Size([1, 16384, 16, 64]), 'stride': (16777216, 1024, 64, 1), 'numel': 16777216, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20075132223488},) cpu output: ({'shape': torch.Size([1, 16384, 16, 64]), 'stride': (16777216, 1024, 64, 1), 'numel': 16777216, 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739319267392},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.copy_                                 input: {'args': ({'shape': torch.Size([1, 16384, 16, 64]), 'stride': (16777216, 1024, 64, 1), 'numel': 16777216, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076747030528}, {'shape': torch.Size([1, 16384, 16, 64]), 'stride': (16777216, 1024, 64, 1), 'numel': 16777216, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20075132223488})}
OpFallbackHook: torch.Tensor.copy_                                 output: ({'shape': torch.Size([1, 16384, 16, 64]), 'stride': (16777216, 1024, 64, 1), 'numel': 16777216, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076610715648},) cpu output: ({'shape': torch.Size([1, 16384, 16, 64]), 'stride': (16777216, 1024, 64, 1), 'numel': 16777216, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739986939968},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.mul                                   input: {'args': ({'shape': torch.Size([1, 16384, 16, 64]), 'stride': (16777216, 1024, 64, 1), 'numel': 16777216, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076338085888}, {'shape': torch.Size([16384, 1, 64]), 'stride': (64, 64, 1), 'numel': 1048576, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067611840000})}
OpFallbackHook: torch.Tensor.mul                                   output: ({'shape': torch.Size([1, 16384, 16, 64]), 'stride': (16777216, 1024, 64, 1), 'numel': 16777216, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076610715648},) cpu output: ({'shape': torch.Size([1, 16384, 16, 64]), 'stride': (16777216, 1024, 64, 1), 'numel': 16777216, 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739386380352},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.mul                                   input: {'args': ({'shape': torch.Size([1, 16384, 16, 64]), 'stride': (16777216, 1024, 64, 1), 'numel': 16777216, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076405195264}, {'shape': torch.Size([16384, 1, 64]), 'stride': (64, 64, 1), 'numel': 1048576, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067607645184})}
OpFallbackHook: torch.Tensor.mul                                   output: ({'shape': torch.Size([1, 16384, 16, 64]), 'stride': (16777216, 1024, 64, 1), 'numel': 16777216, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076677825024},) cpu output: ({'shape': torch.Size([1, 16384, 16, 64]), 'stride': (16777216, 1024, 64, 1), 'numel': 16777216, 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739386380352},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.add                                   input: {'args': ({'shape': torch.Size([1, 16384, 16, 64]), 'stride': (16777216, 1024, 64, 1), 'numel': 16777216, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076610715648}, {'shape': torch.Size([1, 16384, 16, 64]), 'stride': (16777216, 1024, 64, 1), 'numel': 16777216, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076677825024})}
OpFallbackHook: torch.Tensor.add                                   output: ({'shape': torch.Size([1, 16384, 16, 64]), 'stride': (16777216, 1024, 64, 1), 'numel': 16777216, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20075132223488},) cpu output: ({'shape': torch.Size([1, 16384, 16, 64]), 'stride': (16777216, 1024, 64, 1), 'numel': 16777216, 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739319267392},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.copy_                                 input: {'args': ({'shape': torch.Size([1, 16384, 16, 64]), 'stride': (16777216, 1024, 64, 1), 'numel': 16777216, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076780585472}, {'shape': torch.Size([1, 16384, 16, 64]), 'stride': (16777216, 1024, 64, 1), 'numel': 16777216, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20075132223488})}
OpFallbackHook: torch.Tensor.copy_                                 output: ({'shape': torch.Size([1, 16384, 16, 64]), 'stride': (16777216, 1024, 64, 1), 'numel': 16777216, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076610715648},) cpu output: ({'shape': torch.Size([1, 16384, 16, 64]), 'stride': (16777216, 1024, 64, 1), 'numel': 16777216, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739986939968},) dtype_convert_back_dict:{}
skip OpFallbackHook on torch.Tensor.device.__get__
skip OpFallbackHook on torch.Tensor.device.__get__
skip OpFallbackHook on torch.Tensor.dtype.__get__
skip OpFallbackHook on torch.Tensor.dtype.__get__
skip OpFallbackHook on torch.Tensor.shape.__get__
skip OpFallbackHook on torch.Tensor.shape.__get__
skip OpFallbackHook on torch.Tensor.shape.__get__
OpFallbackHook: torch.Tensor.__getitem__                           input: {'args': ({'shape': torch.Size([1, 16384, 8, 128]), 'stride': (16777216, 1024, 128, 1), 'numel': 16777216, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067720888832}, ('Ellipsis', 'slice(None, 128, None)'))}
OpFallbackHook: torch.Tensor.__getitem__                           output: ({'shape': torch.Size([1, 16384, 8, 128]), 'stride': (16777216, 1024, 128, 1), 'numel': 16777216, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074781999104},) cpu output: ({'shape': torch.Size([1, 16384, 8, 128]), 'stride': (16777216, 1024, 128, 1), 'numel': 16777216, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 33821921408},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.chunk                                 input: {'args': ({'shape': torch.Size([1, 16384, 8, 128]), 'stride': (16777216, 1024, 128, 1), 'numel': 16777216, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074781999104}, '2'), 'kwargs': {'dim': '-1'}}
OpFallbackHook: torch.Tensor.chunk                                 output: (({'shape': torch.Size([1, 16384, 8, 64]), 'stride': (8388608, 512, 64, 1), 'numel': 8388608, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074815554048}, {'shape': torch.Size([1, 16384, 8, 64]), 'stride': (8388608, 512, 64, 1), 'numel': 8388608, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074832331776}),) cpu output: (({'shape': torch.Size([1, 16384, 8, 64]), 'stride': (16777216, 1024, 128, 1), 'numel': 8388608, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 33821921408}, {'shape': torch.Size([1, 16384, 8, 64]), 'stride': (16777216, 1024, 128, 1), 'numel': 8388608, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 33821921536}),) dtype_convert_back_dict:{}
skip OpFallbackHook on None
OpFallbackHook: torch.Tensor.__getitem__                           input: {'args': ({'shape': torch.Size([1, 16384, 8, 128]), 'stride': (16777216, 1024, 128, 1), 'numel': 16777216, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074851205120}, ('Ellipsis', 'slice(None, 128, None)'))}
OpFallbackHook: torch.Tensor.__getitem__                           output: ({'shape': torch.Size([1, 16384, 8, 128]), 'stride': (16777216, 1024, 128, 1), 'numel': 16777216, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074884760064},) cpu output: ({'shape': torch.Size([1, 16384, 8, 128]), 'stride': (16777216, 1024, 128, 1), 'numel': 16777216, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739986939968},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.chunk                                 input: {'args': ({'shape': torch.Size([1, 16384, 8, 128]), 'stride': (16777216, 1024, 128, 1), 'numel': 16777216, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074884760064}, '2'), 'kwargs': {'dim': '-1'}}
OpFallbackHook: torch.Tensor.chunk                                 output: (({'shape': torch.Size([1, 16384, 8, 64]), 'stride': (8388608, 512, 64, 1), 'numel': 8388608, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20074989617152}, {'shape': torch.Size([1, 16384, 8, 64]), 'stride': (8388608, 512, 64, 1), 'numel': 8388608, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20075006394880}),) cpu output: (({'shape': torch.Size([1, 16384, 8, 64]), 'stride': (16777216, 1024, 128, 1), 'numel': 8388608, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739986939968}, {'shape': torch.Size([1, 16384, 8, 64]), 'stride': (16777216, 1024, 128, 1), 'numel': 8388608, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139739986940096}),) dtype_convert_back_dict:{}
skip OpFallbackHook on torch.Tensor.shape.__get__
skip OpFallbackHook on torch.Tensor.shape.__get__
OpFallbackHook: torch.Tensor.reshape                               input: {'args': ({'shape': torch.Size([16384, 64]), 'stride': (64, 1), 'numel': 1048576, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067599254528}, ['16384', '1', '64'])}
OpFallbackHook: torch.Tensor.reshape                               output: ({'shape': torch.Size([16384, 1, 64]), 'stride': (64, 64, 1), 'numel': 1048576, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067603449856},) cpu output: ({'shape': torch.Size([16384, 1, 64]), 'stride': (64, 64, 1), 'numel': 1048576, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 33555245696},) dtype_convert_back_dict:{}
skip OpFallbackHook on torch.Tensor.shape.__get__
skip OpFallbackHook on torch.Tensor.shape.__get__
OpFallbackHook: torch.Tensor.reshape                               input: {'args': ({'shape': torch.Size([16384, 64]), 'stride': (64, 1), 'numel': 1048576, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067601352192}, ['16384', '1', '64'])}
OpFallbackHook: torch.Tensor.reshape                               output: ({'shape': torch.Size([16384, 1, 64]), 'stride': (64, 64, 1), 'numel': 1048576, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067605547520},) cpu output: ({'shape': torch.Size([16384, 1, 64]), 'stride': (64, 64, 1), 'numel': 1048576, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 33717063232},) dtype_convert_back_dict:{}
```
