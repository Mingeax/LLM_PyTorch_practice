import torch.nn.functional as F
import torch

y = torch.tensor([1.0]) # 真实标签
x1 = torch.tensor([1.1]) # 输入特征
w1 = torch.tensor([2.2]) # 权重参数
b = torch.tensor([0.0]) # 偏置单元
z = x1 * w1 + b # 网络输入
a = torch.sigmond(z) # 激活和输出
loss = F.binary_cross_entropy(a,y) # 损失函数
