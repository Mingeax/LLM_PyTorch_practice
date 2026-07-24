# LLM & PyTorch 实践项目

这是一个用于学习大语言模型(LLM)和 PyTorch 框架的实践项目。

## 项目结构

```
LLM_PyTorch_practice/
├── README.md                    # 项目说明文档
├── environment.yml              # Conda 环境配置
├── .gitignore                   # Git 忽略规则
└── src/
    ├── construct_llm/           # LLM 构建模块
    │   ├── __init__.py
    │   ├── tokenizer.py         # 简单分词器实现
    │   └── assets/
    │       └── theVerdict.txt   # 训练文本数据
    └── pytorch_example/         # PyTorch 示例
        ├── __init__.py
        └── neural_network.py    # 神经网络训练示例
```

## 模块说明

### 1. construct_llm - 文本分词器

实现了一个基于简单词汇表的分词器,包含以下功能:
- 文本预处理和词汇表构建
- 文本编码 (text → token IDs)
- 文本解码 (token IDs → text)

**使用方法:**
```bash
python -m src.construct_llm.tokenizer
```

### 2. pytorch_example - PyTorch 神经网络示例

包含完整的 PyTorch 深度学习示例:
- 梯度计算演示 (使用 autograd)
- 神经网络模型定义 (NeuralNetwork)
- 自定义数据集 (ToyDataset)
- 模型训练循环
- 准确率评估
- 模型保存与加载

**主要特性:**
- 多层全连接神经网络
- ReLU 激活函数
- 交叉熵损失函数
- SGD 优化器
- 支持 CUDA/MPS/CPU 自动检测

**使用方法:**
```bash
python -m src.pytorch_example.neural_network
```

## 环境配置

### 使用 Conda (推荐)

```bash
# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate my_pytorch_env
```

### 环境要求

- Python 3.11
- PyTorch 2.5.1 (CUDA 12.1 版本)
- torchvision 0.20.1
- torchaudio 2.5.1

### 硬件要求

- **GPU (推荐)**: NVIDIA GPU with CUDA 12.1
- **Apple Silicon**: MPS (Metal Performance Shaders) 支持
- **CPU**: 也支持,但训练速度较慢

## 运行示例

### 运行分词器

```bash
python -m src.construct_llm.tokenizer
```

输出示例:
```
vocab size: 1137
vocab item: ('--', 0)
vocab item: ('—', 1)
...
ids:  [...]
decode ids:  [...]
```

### 运行神经网络训练

```bash
python -m src.pytorch_example.neural_network
```

输出示例:
```
grad_L_w1, grad_L_b:  ...
w1.grad, b.grad:  ...
weight shape:  torch.Size([30, 2])
Batch 1: ...
Epoch: 001/003 | Batch 0/2 | Train Loss: 0.77
...
compute_accuracy: 1.0
CUDA is available: True
Using device: cuda
```

## 学习目标

1. **分词器实现**: 理解文本预处理和词汇表构建的基本概念
2. **PyTorch 基础**: 学习张量操作、自动微分和计算图
3. **神经网络**: 实现多层感知机(MLP)
4. **数据加载**: 使用 Dataset 和 DataLoader
5. **训练流程**: 理解前向传播、损失计算、反向传播和参数更新
6. **模型部署**: 学习模型保存和加载

## 后续扩展建议

- [ ] 添加单元测试
- [ ] 实现更复杂的分词器 (BPE, WordPiece)
- [ ] 添加更多神经网络架构 (CNN, RNN, Transformer)
- [ ] 集成真实数据集 (MNIST, CIFAR-10)
- [ ] 添加模型评估和可视化工具
- [ ] 实现学习率调度和正则化

## 开发建议

每次完成一个功能, 都要更新本文档和测试用例

## 注意事项

- 神经网络训练结果可能因设备而异
- 某些操作在 MPS (Apple Silicon) 上可能受限,代码会自动降级到 CPU
- 首次运行时会自动下载依赖并编译模型

## License

MIT License