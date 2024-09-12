# ditorch

ditorch 是设备无关 torch， 旨在屏蔽各硬件厂商 torch 差异，为用户提供一致使用体验。通过 ditorch，开发者可以适配多个硬件算子库；此外，ditorch 提供训练过程中需要的基础工具，解决模型训练过程中出现的痛点问题。


![ditorch 结构图](ditorch.png)

# **核心功能**
## **1. 可无感切换 pytorch 至国产芯片**

只需添加两行代码，即可在国产芯片上像官方 pytorch 一样使用。
```
import torch
import ditorch
```

## **2. 提供多个基础工具，解决训练过程的问题**

提供模型训练过程中需要的基础工具，解决模型训练过程中出现的痛点问题 [算子工具](op_tools/README.md)。

| 序号 |  工具  |  简介  |
| ---- |  ------  |  ------  |
| 1 | [算子参数抓取工具](#tool1) |  抓取模型真实训练过程中真实的输入输出  |
| 2 | [精度分析工具](#tool2) | 进行离线和实时的精度分析 |
| 3 | [速度分析工具](#tool3) | 可进行离线和实时的耗时分析，协助性能优化 |
| 4 | [算子 Fallback](#tool4) | 可将指定、全部算子在设备上运行的操作 fallback 到 CPU 计算 |
| 5 | [算子数据类型转换工具](#tool5) | 可将指定、全部算子的特定数据类型转到给定数据类型去计算 |


### **算子参数抓取工具** <a id="tool1"></a>
抓取模型真实训练过程中真实的输入输出：
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

#### **抓取前向和反向的所有输入输出**

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

#### **只抓取sort算子的参数，忽略其他算子 OP_CAPTURE_LIST=torch.Tensor.sort**
```
...
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

#### **排除指定算子，抓取所有其他算子 OP_CAPTURE_DISABLE_LIST="torch.Tensor.add,torch.Tensor.sub"**
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
### **精度分析工具** <a id="tool2"></a>
精度分析工具可以实现：
1. 离线分析：用模型训练过程中真实输入输出，离线对比。
2. 实时精度对比：模型训练时实时与cpu对比分析精度。

```
# usage1
import op_tools
with op_tools.OpAutoCompare():
    code_snippet_to_autocompare()
```

```
# usage2
import op_tools
autocompare = op_tools.OpAutoCompare()
autocompare.start()
code_snippet_to_autocompare()
autocompare.stop()
```
可通过设置: AUTOCOMPARE_ERROR_TOLERANCE_FLOAT16,  AUTOCOMPARE_ERROR_TOLERANCE_FLOAT32,  AUTOCOMPARE_ERROR_TOLERANCE_FLOAT64,  AUTOCOMPARE_ERROR_TOLERANCE 这几个环境变量来自定义精度阈值。

```
# for float16
export AUTOCOMPARE_ERROR_TOLERANCE_FLOAT16="1e-3,1e-4" # atol=1e-3, rtol=1e-4
# for bfloat16
export AUTOCOMPARE_ERROR_TOLERANCE_BFLOAT16="1e-2,1e-3" # atol=1e-2, rtol=1e-3
# for other dtype
export AUTOCOMPARE_ERROR_TOLERANCE="1e-4,1e-5" # atol=1e-4, rtol=1e-5
```

#### **基于InternEvo + ditorch + torch_npu 在华为910B上实时精度分析输出片段**


```
...
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
...
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
OpAutoCompareHook: torch.Tensor.sum (ins[0].grad)                     allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.sort (ins[0].grad)                    allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.add (ins[0].grad)                     allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.add (ins[1].grad)                     allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.half (ins[0].grad)                    allclose: True    max_diff:          0.000000000
...
OpAutoCompareHook: torch.nn.functional.silu (ins[0].grad)             allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.nn.functional.linear (ins[2].grad)           allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.nn.functional.linear (ins[1].grad)           allclose: True    max_diff:          0.000000238
OpAutoCompareHook: torch.nn.functional.linear (ins[0].grad)           allclose: True    max_diff:          0.000000060
...
OpAutoCompareHook: torch.Tensor.add_ (ins[0].grad)                    allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.div_ (ins[0].grad)                    allclose: True    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.div (ins[0].grad)                     allclose: True    max_diff:          0.000000000
...
OpAutoCompareHook: torch.mul (ins[0].grad)                            allclose: False   max_diff:          4.077112675
OpAutoCompareHook: torch.mul (ins[1].grad)                            allclose: False   max_diff:          4.077112675
```

#### **离线算子精度测试**
```
python op_tools/run_op_from_data.py /deeplink/op_capture_result/torch.Tensor.div/2334011/5  --acc_check --run_times 1
ditorch.framework: torch_npu:2.1.0.post3
OpAutoCompareHook: torch.Tensor.div                                   allclose: True    max_diff:          0.000000060
OpAutoCompareHook: torch.Tensor.div 0th input grad                    allclose: True    max_diff:          0.000000954
OpAutoCompareHook: torch.Tensor.div 1th input grad                    allclose: True    max_diff:          0.000000238
```

### 速度分析工具 <a id="tool3"> </a>

速度分析工具同样可以支持（1）离线分析和（2）实时分析。

用模型训练过程中真实的输入输出分析算子和通信的耗时，分析出性能瓶颈
```
# 测量算子耗时（输入为使用算子抓取工具在模型训练时抓取到的真实数据）
python op_tools/run_op_from_data.py /deeplink/op_capture_result/torch.Tensor.div/2334011/5 --run_times 3 --sync_time_measure
ditorch.framework: torch_npu:2.1.0.post3
SyncExecuteTimer: torch.Tensor.div forward  elasped 69.61202621 ms
SyncExecuteTimer: torch.Tensor.div backward elasped 169.42977905 ms
SyncExecuteTimer: torch.Tensor.div forward  elasped 0.08678436 ms
SyncExecuteTimer: torch.Tensor.div backward elasped 2.97260284 ms
SyncExecuteTimer: torch.Tensor.div forward  elasped 0.04935265 ms
SyncExecuteTimer: torch.Tensor.div backward elasped 0.16641617 ms
```

#### **只跑指定算子3遍前向**
```
ditorch/op_tools# python run_op_from_data.py /op_capture_result/torch.Tensor.div/2278281/5  --run_times 3 --only_run_forward --sync_time_measure
ditorch.framework: torch_npu:2.1.0.post3
/deeplink_afs/zhaoguochun/ditorch/op_tools/op_capture_result/torch.Tensor.div/2278281/5
SyncExecuteTimer: torch.Tensor.div forward elasped 91.06540680 ms
SyncExecuteTimer: torch.Tensor.div forward elasped 0.24318695 ms
SyncExecuteTimer: torch.Tensor.div forward elasped 0.07224083 ms
```

#### **模型训练时算子耗时分析 (前向 + 反向)**
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
OpTimeMeasureHook: torch.cat                      forward elasped:  0.02408028 ms     input: {'args': ([{'shape': torch.Size([1]), 'stride': (1,), 'numel': 1, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067179863040}, {'shape': torch.Size([2]), 'stride': (1,), 'numel': 2, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067179862016}],)} output: {'args': ({'shape': torch.Size([3]), 'stride': (1,), 'numel': 3, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067179862528},)}
OpTimeMeasureHook: torch.Tensor.add_              forward elasped:  0.02861023 ms     input: {'args': ({'shape': torch.Size([3]), 'stride': (1,), 'numel': 3, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067179863552}, {'shape': torch.Size([3]), 'stride': (1,), 'numel': 3, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067180651008})} output: {'args': ({'shape': torch.Size([3]), 'stride': (1,), 'numel': 3, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067179863552},)}
OpTimeMeasureHook: torch.Tensor.mul               forward elasped:  0.03147125 ms     input: {'args': ({'shape': torch.Size([1]), 'stride': (1,), 'numel': 1, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067179860480}, {'shape': torch.Size([]), 'stride': (), 'numel': 1, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': True, 'layout': 'torch.strided', 'data': 20067179859456})} output: {'args': ({'shape': torch.Size([1]), 'stride': (1,), 'numel': 1, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': True, 'layout': 'torch.strided', 'data': 20067179862016},)}
...
OpTimeMeasureHook: torch.Tensor.add_              backward elasped: 0.01120567 ms     grad_inputs: {'args': (({'shape': torch.Size([]), 'stride': (), 'numel': 1, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067180634112}, 'None'),)} output: {'args': (({'shape': torch.Size([]), 'stride': (), 'numel': 1, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067180634112},),)}
OpTimeMeasureHook: torch.Tensor.div_              backward elasped: 0.03290176 ms     grad_inputs: {'args': (({'shape': torch.Size([]), 'stride': (), 'numel': 1, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067179863040}, 'None'),)} output: {'args': (({'shape': torch.Size([]), 'stride': (), 'numel': 1, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067180634112},),)}
OpTimeMeasureHook: torch.Tensor.div               backward elasped: 0.06675720 ms     grad_inputs: {'args': (({'shape': torch.Size([]), 'stride': (), 'numel': 1, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067180634112}, 'None'),)} output: {'args': (({'shape': torch.Size([]), 'stride': (), 'numel': 1, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067179863040},),)}
OpTimeMeasureHook: torch.Tensor.sum               backward elasped: 0.01549721 ms     grad_inputs: {'args': (({'shape': torch.Size([16384]), 'stride': (0,), 'numel': 16384, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067180634112},),)} output: {'args': (({'shape': torch.Size([]), 'stride': (), 'numel': 1, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067180634112},),)}
OpTimeMeasureHook: torch.nn.functional.cross_entropy backward elasped: 4.20713425 ms     grad_inputs: {'args': (({'shape': torch.Size([16384, 92544]), 'stride': (92544, 1), 'numel': 1516240896, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20095078236160},),)} output: {'args': (({'shape': torch.Size([16384]), 'stride': (0,), 'numel': 16384, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067180634112},),)}
OpTimeMeasureHook: torch.Tensor.float             backward elasped: 7.45630264 ms     grad_inputs: {'args': (({'shape': torch.Size([1, 16384, 92544]), 'stride': (1516240896, 92544, 1), 'numel': 1516240896, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20091857010688},),)} output: {'args': (({'shape': torch.Size([1, 16384, 92544]), 'stride': (1516240896, 92544, 1), 'numel': 1516240896, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': True, 'layout': 'torch.strided', 'data': 20107963138048},),)}
OpTimeMeasureHook: torch.nn.functional.normalize  backward elasped: 5.66196442 ms     grad_inputs: {'args': (({'shape': torch.Size([92544, 2048]), 'stride': (2048, 1), 'numel': 189530112, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20090863111168}, {'shape': torch.Size([92544, 2048]), 'stride': (2048, 1), 'numel': 189530112, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20086551741952}),)} output: {'args': (({'shape': torch.Size([92544, 2048]), 'stride': (2048, 1), 'numel': 189530112, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20085414559744},),)}
```

### 算子 fallback <a id="tool4"> </a>
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

#### **只 fallback 指定算子 (export OP_FALLBACK_LIST="torch.nn.functional.linear")**
```
skip OpFallbackHook on torch.Tensor.float
skip OpFallbackHook on torch.Tensor.add
skip OpFallbackHook on torch.Tensor.div
skip OpFallbackHook on torch.Tensor.item
skip OpFallbackHook on torch.Tensor.float
skip OpFallbackHook on torch.Tensor.div
skip OpFallbackHook on torch.Tensor.fill_
skip OpFallbackHook on torch.Tensor.is_complex
skip OpFallbackHook on torch.Tensor.numel
skip OpFallbackHook on torch.Tensor.unbind
skip OpFallbackHook on torch.Tensor.sub
skip OpFallbackHook on torch.Tensor.max
...
OpFallbackHook: torch.nn.functional.linear                         input: {'args': ({'shape': torch.Size([1, 16384, 2048]), 'stride': (33554432, 2048, 1), 'numel': 33554432, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20076203868160}, {'shape': torch.Size([4096, 2048]), 'stride': (2048, 1), 'numel': 8388608, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': True, 'layout': 'torch.strided', 'data': 20077985398784}, 'None')}
OpFallbackHook: torch.nn.functional.linear                         output: ({'shape': torch.Size([1, 16384, 4096]), 'stride': (67108864, 4096, 1), 'numel': 67108864, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20075820089344},) cpu output: ({'shape': torch.Size([1, 16384, 4096]), 'stride': (67108864, 4096, 1), 'numel': 67108864, 'dtype': 'torch.bfloat16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 139743270527040},) dtype_convert_back_dict:{}
skip OpFallbackHook on torch.Tensor.shape.__get__
...
```

#### **fallback 指定算子以外所有算子（export OP_FALLBACK_DISABLE_LIST="torch.nn.functional.linear"）**
```
...
skip OpFallbackHook on torch.nn.functional.linear
OpFallbackHook: torch.Tensor.float                                 input: {'args': ({'shape': torch.Size([1, 16384, 92544]), 'stride': (1516240896, 92544, 1), 'numel': 1516240896, 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'requires_grad': True, 'layout': 'torch.strided', 'data': 20081119592448},)}
OpFallbackHook: torch.Tensor.float                                 output: ({'shape': torch.Size([1, 16384, 92544]), 'stride': (1516240896, 92544, 1), 'numel': 1516240896, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': True, 'layout': 'torch.strided', 'data': 20100446945280},) cpu output: ({'shape': torch.Size([1, 16384, 92544]), 'stride': (1516240896, 92544, 1), 'numel': 1516240896, 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': True, 'layout': 'torch.strided', 'data': 140152888873024},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.contiguous                            input: {'args': ({'shape': torch.Size([1, 16384, 92544]), 'stride': (1516240896, 92544, 1), 'numel': 1516240896, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': True, 'layout': 'torch.strided', 'data': 20100446945280},)}
OpFallbackHook: torch.Tensor.contiguous                            output: ({'shape': torch.Size([1, 16384, 92544]), 'stride': (1516240896, 92544, 1), 'numel': 1516240896, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': True, 'layout': 'torch.strided', 'data': 20106889396224},) cpu output: ({'shape': torch.Size([1, 16384, 92544]), 'stride': (1516240896, 92544, 1), 'numel': 1516240896, 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': True, 'layout': 'torch.strided', 'data': 140155921358912},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.view                                  input: {'args': ({'shape': torch.Size([1, 16384, 92544]), 'stride': (1516240896, 92544, 1), 'numel': 1516240896, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': True, 'layout': 'torch.strided', 'data': 20106889396224}, '-1', '92544')}
OpFallbackHook: torch.Tensor.view                                  output: ({'shape': torch.Size([16384, 92544]), 'stride': (92544, 1), 'numel': 1516240896, 'dtype': 'torch.float32', 'device': 'npu:0', 'requires_grad': True, 'layout': 'torch.strided', 'data': 20113331847168},) cpu output: ({'shape': torch.Size([16384, 92544]), 'stride': (92544, 1), 'numel': 1516240896, 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': True, 'layout': 'torch.strided', 'data': 140155921358912},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.contiguous                            input: {'args': ({'shape': torch.Size([1, 16384]), 'stride': (16384, 1), 'numel': 16384, 'dtype': 'torch.int64', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067180535808},)}
OpFallbackHook: torch.Tensor.contiguous                            output: ({'shape': torch.Size([1, 16384]), 'stride': (16384, 1), 'numel': 16384, 'dtype': 'torch.int64', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067179877888},) cpu output: ({'shape': torch.Size([1, 16384]), 'stride': (16384, 1), 'numel': 16384, 'dtype': 'torch.int64', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 33663304832},) dtype_convert_back_dict:{}
OpFallbackHook: torch.Tensor.view                                  input: {'args': ({'shape': torch.Size([1, 16384]), 'stride': (16384, 1), 'numel': 16384, 'dtype': 'torch.int64', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067179877888}, '-1')}
...

```

#### **fallback 所有算子时部分输出**
```
...
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
...
```

### **算子数据类型转换工具** <a id="tool5"></a>

```
# usage1
export OP_DTYPE_CAST_DICT="torch.float16->torch.float32,torch.bfloat16->torch.float32"
with op_tools.OpDtypeCast():
    f()

# usage2
dtype_caster = op_tools.OpDtypeCast()
dtype_caster.start()
for i in range(3):
    f()
dtype_caster.stop()
```

```
# usage3
os.environ["OP_DTYPE_CAST_DISABLE_LIST"] = "torch.Tensor.add,torch.Tensor.sub"
dtype_caster.start()
f()
dtype_caster.stop()
```
```
# usage4
os.environ["OP_DTYPE_CAST_DISABLE_LIST"] = ""
os.environ["OP_DTYPE_CAST_LIST"] = "torch.Tensor.sort,torch.Tensor.add"  # only cast these op
os.environ["OP_DTYPE_CAST_DICT"] = "torch.half->torch.bfloat16"
dtype_caster.start()
f()
dtype_caster.stop()
```

```
apply OpDtypeCastHook on torch.nn.functional.linear
OpDtypeCastHook: torch.nn.functional.linear                         0th arg torch.float16 -> torch.float32  config:torch.float16->torch.float32,torch.bfloat16->torch.float32
OpDtypeCastHook: torch.nn.functional.linear                         1th arg torch.float16 -> torch.float32  config:torch.float16->torch.float32,torch.bfloat16->torch.float32
OpDtypeCastHook: torch.nn.functional.linear                         2th arg torch.float16 -> torch.float32  config:torch.float16->torch.float32,torch.bfloat16->torch.float32
OpDtypeCastHook: torch.nn.functional.linear                         0th out torch.float32 -> torch.float16  config:torch.float16->torch.float32,torch.bfloat16->torch.float32
apply OpDtypeCastHook on torch.Tensor.add
OpDtypeCastHook: torch.Tensor.add                                   0th arg torch.float16 -> torch.float32  config:torch.float16->torch.float32,torch.bfloat16->torch.float32
OpDtypeCastHook: torch.Tensor.add                                   1th arg torch.float16 -> torch.float32  config:torch.float16->torch.float32,torch.bfloat16->torch.float32
OpDtypeCastHook: torch.Tensor.add                                   0th out torch.float32 -> torch.float16  config:torch.float16->torch.float32,torch.bfloat16->torch.float32
apply OpDtypeCastHook on torch.Tensor.sub
OpDtypeCastHook: torch.Tensor.sub                                   0th arg torch.float16 -> torch.float32  config:torch.float16->torch.float32,torch.bfloat16->torch.float32
OpDtypeCastHook: torch.Tensor.sub                                   1th arg torch.float16 -> torch.float32  config:torch.float16->torch.float32,torch.bfloat16->torch.float32
OpDtypeCastHook: torch.Tensor.sub                                   0th out torch.float32 -> torch.float16  config:torch.float16->torch.float32,torch.bfloat16->torch.float32
apply OpDtypeCastHook on torch.Tensor.div
OpDtypeCastHook: torch.Tensor.div                                   0th arg torch.float16 -> torch.float32  config:torch.float16->torch.float32,torch.bfloat16->torch.float32
OpDtypeCastHook: torch.Tensor.div                                   1th arg torch.float16 -> torch.float32  config:torch.float16->torch.float32,torch.bfloat16->torch.float32
OpDtypeCastHook: torch.Tensor.div                                   0th out torch.float32 -> torch.float16  config:torch.float16->torch.float32,torch.bfloat16->torch.float32
apply OpDtypeCastHook on torch.Tensor.sort
OpDtypeCastHook: torch.Tensor.sort                                  0th arg torch.float16 -> torch.float32  config:torch.float16->torch.float32,torch.bfloat16->torch.float32
OpDtypeCastHook: torch.Tensor.sort                                  0th out torch.float32 -> torch.float16  config:torch.float16->torch.float32,torch.bfloat16->torch.float32
apply OpDtypeCastHook on torch.Tensor.__getitem__
OpDtypeCastHook: torch.Tensor.__getitem__                           0th arg torch.float16 -> torch.float32  config:torch.float16->torch.float32,torch.bfloat16->torch.float32
OpDtypeCastHook: torch.Tensor.__getitem__                           0th out torch.float32 -> torch.float16  config:torch.float16->torch.float32,torch.bfloat16->torch.float32
apply OpDtypeCastHook on torch.Tensor.sum
OpDtypeCastHook: torch.Tensor.sum                                   0th arg torch.float16 -> torch.float32  config:torch.float16->torch.float32,torch.bfloat16->torch.float32
OpDtypeCastHook: torch.Tensor.sum                                   0th out torch.float32 -> torch.float16  config:torch.float16->torch.float32,torch.bfloat16->torch.float32
```

### 自定义算子工具生效的条件
```
def apply_feature(ops, feature, condition_func=lambda *args, **kwargs: True):
    ...
```

op_tools.apply_feature接口可以作用在torch接口和其他第三方接口上，通过condition_func参数可以自定义生效条件，当condition_func返回True时，工具生效，否则不生效。condition_func的输入形参和算子输入形参相同。
feature参数为功能特性，目前支持以下类型：
- fallback: 算子fallback
- cast_dtype: 算子数据类型转换
- op_capture: 算子参数抓取
- autocompare: 算子精度对比  (做精度对比时，需要设备实现和cpu实现的调用接口一致)
- dump_op_args: 算子参数打印
- measure_op_time: 算子执行时间测量


```
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
```

#### 1.按需fallback
```
op_tools.apply_feature("torch.add", feature="fallback", condition_func=custom_condition)
torch.add(x, x)
```
outputs:
```
hook enable because a.dtype is float16
apply OpFallbackHook on torch.add
OpFallbackHook: torch.add                                          input: {'args': ({'shape': torch.Size([2, 3, 4]), 'stride': (12, 4, 1), 'numel': 24, 'dtype': 'torch.float16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067179823104}, {'shape': torch.Size([2, 3, 4]), 'stride': (12, 4, 1), 'numel': 24, 'dtype': 'torch.float16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067179823104})}
OpFallbackHook: torch.add                                          output: ({'shape': torch.Size([2, 3, 4]), 'stride': (12, 4, 1), 'numel': 24, 'dtype': 'torch.float16', 'device': 'npu:0', 'requires_grad': False, 'layout': 'torch.strided', 'data': 20067179824640},) cpu output: ({'shape': torch.Size([2, 3, 4]), 'stride': (12, 4, 1), 'numel': 24, 'dtype': 'torch.float16', 'device': 'cpu', 'requires_grad': False, 'layout': 'torch.strided', 'data': 544920640},) dtype_convert_back_dict:{}
```

#### 2.按需autocompare
```
op_tools.apply_feature("torch.sub", feature="autocompare", condition_func=custom_condition)
torch.sub(y, y)
torch.sub(z, z)
```
output:
```
hook enable because a.dim() is 2
apply OpAutoCompareHook on torch.sub
compare_result: torch.sub                                            allclose: True     max_abs_diff:          0.000000000      max_relative_diff:          0.000000000

hook disable
skip OpAutoCompareHook on torch.sub
```

#### 3.抓取特定输入情况下的算子输入输出
```
op_tools.apply_feature("torch.mul", feature="op_capture", condition_func=custom_condition)
torch.mul(x, x)
```
output:
```
hook enable because a.dtype is float16
apply OpCaptureHook on torch.mul
op_capture_result/torch.mul/366650/3/input.pth saved
op_capture_result/torch.mul/366650/3/output.pth saved
```

#### 4.将特定算子的输入数据类型转成特定数据类型计算
```
op_tools.apply_feature("torch.div", feature="cast_dtype", condition_func=custom_condition)
os.environ["OP_DTYPE_CAST_DICT"] = "float32->float16"
torch.div(y, y)
```
output:
```
hook enable because a.dim() is 2
apply OpDtypeCastHook on torch.div
OpDtypeCastHook: torch.div                                          0th arg torch.float32 -> torch.float16  config:torch.float32->torch.float16
OpDtypeCastHook: torch.div                                          1th arg torch.float32 -> torch.float16  config:torch.float32->torch.float16
OpDtypeCastHook: torch.div                                          0th out torch.float16 -> torch.float32  config:torch.float32->torch.float16
```

### 相关环境变量
|               工具               |                 环境变量名                    |                        值                                  |          说明             |    备注                      |
|---------------------------------|----------------------------------------------|-----------------------------------------------------------|---------------------------|-----------------------------|
|  [算子参数抓取工具](#tool1)        | OP_CAPTURE_DISABLE_LIST                      | torch.add,torch.nn.functional.linear,torch.Tensor.relu_   | 不抓取这些算子的参数         | 算子名全称，多个算子时以逗号隔开  |
|  [算子参数抓取工具](#tool1)        | OP_CAPTURE_LIST                              |                       同上                                 | 只抓取这些算子的参数         |           同上               |
|  [精度分析工具](#tool2)           | OP_AUTOCOMPARE_LIST                          |                       同上                                 | 只对指定的算子做精度对比      |           同上               |
|  [精度分析工具](#tool2)           | OP_AUTOCOMPARE_DISABLE_LIST                  |                       同上                                 | 精度对比时忽略指定的这些算子   |           同上               |
|  [算子数据类型转换工具](#tool5)    | OP_DTYPE_CAST_DISABLE_LIST                   |                       同上                                 | 做类型转换时忽略指定的这些算子 |           同上               |
|  [算子数据类型转换工具](#tool5)    | OP_DTYPE_CAST_LIST                           |                       同上                                 | 只对指定的算子做类型转换      |           同上               |
|  [精度分析工具](#tool2)           | AUTOCOMPARE_ERROR_TOLERANCE                  |                      atol,rtol                             | allclose 参数             | 如设置，则使用给定的误差阈阈值覆盖默认值    |
|  [精度分析工具](#tool2)           | AUTOCOMPARE_ERROR_TOLERANCE_FLOAT16          |                      atol,rtol                             | allclose 参数             | 如设置且数据类型满足，则使用给定的误差阈值          |
|  [精度分析工具](#tool2)           | AUTOCOMPARE_ERROR_TOLERANCE_BFLOAT16         |                      atol,rtol                             | allclose 参数             |           同上                |
|  [精度分析工具](#tool2)           | AUTOCOMPARE_ERROR_TOLERANCE_FLOAT32          |                      atol,rtol                             | allclose 参数             |           同上                |
|  [精度分析工具](#tool2)           | AUTOCOMPARE_ERROR_TOLERANCE_FLOAT64          |                      atol,rtol                             | allclose 参数             |           同上                |
|  [精度分析工具](#tool2)           | LINEAR_AUTOCOMPARE_ERROR_TOLERANCE_FLOAT16   |                      atol,rtol                             | allclose 参数             |如设置且算子名和数据类型满足，则使用给定的误差阈值。算子名取算子全称最后一个'.'右边的部分，如torch.add,则算子名为ADD_,torch.nn.functional.linear的算子名为LINEAR_                 |
|  [算子数据类型转换工具](#tool5)    | OP_DTYPE_CAST_DICT                           |torch.float16->torch.float32,torch.bfloat16->torch.float32     | 给定要转换的数据类型和目标数据类型    |          有多组时以逗号隔开     |

