import torch.nn.functional as F
from torch.autograd import grad
import torch


y = torch.tensor([1.0])  # 真实标签
x1 = torch.tensor([1.1])  # 输入特征

# 一个节点的requires_grad设为真后, 会在内部构建一个计算图, 便于计算梯度
w1 = torch.tensor([2.2], requires_grad=True)  # 权重参数
b = torch.tensor([0.0], requires_grad=True)  # 偏置单元
z = x1 * w1 + b  # 网络输入
a = torch.sigmoid(z)  # 激活和输出

loss = F.binary_cross_entropy(a, y)  # 损失函数

grad_L_w1 = grad(loss, w1, retain_graph=True)
grad_L_b = grad(loss, b, retain_graph=True)

print(grad_L_w1, grad_L_b)
print("----------")
loss.backward()
print(w1.grad, b.grad)
