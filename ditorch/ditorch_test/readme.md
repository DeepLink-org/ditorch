# 1. <font color=#0099ff>介绍</font>
1. 本工具是`ditorch`的测试工具，用于在不同的国产设备上运行`PyTorch`官方的测试用例。
2. 目前只适配了`torch_npu`。

# 2. <font color=#0099ff>适配逻辑</font>
由于`PyTorch`官方的测试用例主要是针对CUDA设备的，所以需要一些改动才能适配国产设备。 \
通过运行如下命令，即可完成对PyTorch测试脚本的处理。
```
python main.py
```
main.py主要进行以下工作：\
a. 克隆`PyTorch`官方仓库(以下称为origin_torch) \
b. 过滤部分origin_torch中不必要的测试脚本，并将剩余的测试脚本拷贝到processed_tests路径下 \
c. 对processed_tests路径中的测试脚本进行处理

# 3. <font color=#0099ff>测试方法</font>
使用的测试框架是`unittest`，支持以下的命令行选项：
1. <font color=#0099ff>**执行单个测试脚本**</font> \
以test_nn.py为例 
```
python test_nn.py
```
2. <font color=#0099ff>**测试单个测例**</font> \
可以通过以下命令来运行某个特定的测例
```
python -m unittest test_file.Test_Class.test_method
```

3. <font color=#0099ff>**-k**</font> \
只运行匹配模式或子字符串的测试方法和类。 \
以test_autocast.py为例：
```
python test_autocast.py -v -k test_autocast_nn_fp32
```
4. <font color=#0099ff>**-f --failfast**</font> \
当出现第一个错误或者失败时，停止运行测试。

# 4. <font color=#0099ff>错误记录和跳过</font>
1. <font color=#0099ff>**测试Error和Failure自动记录**</font> \
测试结果为Error和Failure的测例会被自动记录到failed_tests_record下的json文件 \
记录格式为：\
`"{test_name}": ["{Error type}", ["{error reason}"]]` \

2. <font color=#0099ff>**根据DISABLED_TESTS_FILE跳过测例**</font>
```
export DISABLED_TESTS_FILE=./unsupported_test_cases/.pytorch-disabled-tests.json
```
如果不是在test目录下运行测试用例，需要传入.pytorch-disabled-tests.json的绝对路径。