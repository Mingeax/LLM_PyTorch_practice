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

# NeuralNetwork 是 torch.nn.Module 子类, 继承了它
class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

		# Sequential类可以使按特定顺序执行每个层更方便
		# 将layer属性设为Sequential实例后, 可以在forward方法调用layers, 而无需单独调用每层
        self.layers = torch.nn.Sequential(
                
            # 第一个隐藏层
            # 线性层
            torch.nn.Linear(num_inputs, 30),
            # 非线性激活函数
            torch.nn.ReLU(),

		    # 前一层的输出是后一层的输入
            # 第二个隐藏层
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # 第三个隐藏层
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits

# 实例化一个新的神经网络对象
model = NeuralNetwork(50, 3)

# 访问权重参数矩阵
model.layers[0].weight

print('weigth: ', model.layers[0].weight.shape)
