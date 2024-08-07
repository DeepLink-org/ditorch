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
# 抓取前向和反向的所有输入输出
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

```
# 只抓取sort算子的参数，忽略其他算子 OP_CAPTURE_LIST=torch.Tensor.sort
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

```
# 排除指定算子，抓取所有其他算子 OP_CAPTURE_DISABLE_LIST="torch.Tensor.add,torch.Tensor.sub"
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
```
# 基于InternEvo + ditorch + torch_npu 在华为910B上实时精度分析输出片段
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

```
# 只跑指定算子3遍前向
ditorch/op_tools# python run_op_from_data.py /op_capture_result/torch.Tensor.div/2278281/5  --run_times 3 --only_run_forward True
ditorch.framework: torch_npu:2.1.0.post3
/deeplink_afs/zhaoguochun/ditorch/op_tools/op_capture_result/torch.Tensor.div/2278281/5
SyncExecuteTimer: torch.Tensor.div forward elasped 91.06540680 ms
SyncExecuteTimer: torch.Tensor.div forward elasped 0.24318695 ms
SyncExecuteTimer: torch.Tensor.div forward elasped 0.07224083 ms
```

```
#模型训练时算子耗时分析 (前向 + 反向)
```