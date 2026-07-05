import torch.nn.functional as F
from torch.autograd import grad
import torch
from torch.utils.data import Dataset, DataLoader


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


# 一个全连接神经网络示例
# class NeuralNetwork(torch.nn.Module):
#     def __init__(self, num_inputs, num_outputs):
#         super().__init__()

#         self.layers = torch.nn.Sequential(

#             # 1st hidden layer
#             torch.nn.Linear(num_inputs, 30),
#             torch.nn.ReLU(),

#             # 2nd hidden layer
#             torch.nn.Linear(30, 20),
#             torch.nn.ReLU(),

#             # output layer
#             torch.nn.Linear(20, num_outputs),
#         )

#     def forward(self, x):
#         logits = self.layers(x)
#         return logits

#  一个小的示例数据集
# 五个训练示例, 每个示例有两个特征
X_train = torch.tensor(
    [
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5],
    ]
)
# 包含相应类别标签的张量, 三个示例属于类别标签0, 两个属于1
Y_train = torch.tensor([0, 0, 0, 1, 1])

# 包含两个样本的测试集
X_test = torch.tensor([[-0.8, 2.8], [2.6, -1.6]])
Y_test = torch.tensor([0, 1])


# 继承Dataset父类来创建一个自定义数据集类ToyDataset
class ToyDataset(Dataset):
    def __init__(self, X, Y):
        self.features = X
        self.labels = Y

    # 检索一条数据记录及其对应标签的说明
    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    # 返回数据集总长度的说明
    def __len__(self):
        return self.labels.shape[0]


train_ds = ToyDataset(X_train, Y_train)
test_ds = ToyDataset(X_test, Y_test)

# 自定义Dataset类后, 可以使用pt的DataLoader类从中进行采样

torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_ds,  # 已有的训练数据集
    batch_size=2,
    shuffle=True,  # 是否打乱数据
    num_workers=0,  # 后台进程数量,设为零则数据加载只在主进程进行,大于零则可在多个工作进程并行加载
    drop_last=True,  # 每轮丢弃最后一个批次(只包含一个实例), 避免影响训练过程中的收敛
)
test_loader = DataLoader(
    dataset=test_ds,
    batch_size=2,
    shuffle=False,  # 测试数据集无需打乱
    num_workers=0,
)

# 实例化数据加载器后, 可对其进行迭代, 这里省略了具体细节
for idx, (x, y) in enumerate(train_loader):
    print(f"Batch {idx + 1}:", x, y)




