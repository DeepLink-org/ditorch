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

#### 精度分析工具
离线分析 + 实时精度对比
1. 用模型训练过程中真实输入输出，离线对比
2. 模型训练时实时与cpu对比分析精度
```
# 基于InternEvo + ditorch + torch_npu 在华为910B上实时精度分析输出片段
OpAutoCompareHook: torch.Tensor.contiguous                            max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.is_complex                            max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.add                                   max_diff:          0.000000000
OpAutoCompareHook: torch.nn.functional.dropout                        max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.to                                    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.to                                    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.pow                                   max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.mean                                  max_diff:          0.000000179
OpAutoCompareHook: torch.Tensor.add                                   max_diff:          0.000000000
OpAutoCompareHook: torch.rsqrt                                        max_diff:          0.000000119
OpAutoCompareHook: torch.Tensor.mul                                   max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.to                                    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.mul                                   max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.contiguous                            max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.contiguous                            max_diff:          0.000000000
OpAutoCompareHook: torch.nn.functional.linear                         max_diff:          0.015625000
op_capture_result/2024-08-02--16-31/1915529/torch.nn.functional.linear/device_input.pth saved
op_capture_result/2024-08-02--16-31/1915529/torch.nn.functional.linear/device_input.pth.json saved
op_capture_result/2024-08-02--16-31/1915529/torch.nn.functional.linear/device_output.pth saved
op_capture_result/2024-08-02--16-31/1915529/torch.nn.functional.linear/device_output.pth.json saved
op_capture_result/2024-08-02--16-31/1915529/torch.nn.functional.linear/cpu_input.pth saved
op_capture_result/2024-08-02--16-31/1915529/torch.nn.functional.linear/cpu_input.pth.json saved
op_capture_result/2024-08-02--16-31/1915529/torch.nn.functional.linear/cpu_output.pth saved
op_capture_result/2024-08-02--16-31/1915529/torch.nn.functional.linear/cpu_output.pth.json saved
OpAutoCompareHook: torch.stack                                        max_diff:          0.000000000
OpAutoCompareHook: torch.functional.norm                              max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.unsqueeze                             max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.__pow__                               max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.to                                    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.to                                    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.to                                    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.float                                 max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.float                                 max_diff:          0.000000000
OpAutoCompareHook: torch.functional.norm                              max_diff:          0.000000000
OpAutoCompareHook: torch.stack                                        max_diff:          0.000000000
OpAutoCompareHook: torch.functional.norm                              max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.unsqueeze                             max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.__pow__                               max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.to                                    max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.float                                 max_diff:          0.000000000
OpAutoCompareHook: torch.Tensor.float                                 max_diff:          0.000000000
OpAutoCompareHook: torch.functional.norm                              max_diff:        760.125000000
op_capture_result/2024-08-02--16-29/1915529/torch.functional.norm/device_input.pth saved
op_capture_result/2024-08-02--16-29/1915529/torch.functional.norm/device_input.pth.json saved
op_capture_result/2024-08-02--16-29/1915529/torch.functional.norm/device_output.pth saved
op_capture_result/2024-08-02--16-29/1915529/torch.functional.norm/device_output.pth.json saved
op_capture_result/2024-08-02--16-29/1915529/torch.functional.norm/cpu_input.pth saved
op_capture_result/2024-08-02--16-29/1915529/torch.functional.norm/cpu_input.pth.json saved
op_capture_result/2024-08-02--16-29/1915529/torch.functional.norm/cpu_output.pth saved
op_capture_result/2024-08-02--16-29/1915529/torch.functional.norm/cpu_output.pth.json saved
OpAutoCompareHook: torch.Tensor.to                                    compare_result: Inconsistent dtypes: torch.float32 torch.float64, max_diff:0.0
```

#### 性能分析工具
用模型训练过程中真实的输入输出分析算子和通信的耗时，分析出性能瓶颈