import torch
import torch.nn.functional as F
from torch.autograd import grad
from torch.utils.data import DataLoader, Dataset

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

print("grad_L_w1, grad_L_b: ", grad_L_w1, grad_L_b)
print("----------")
loss.backward()
print("w1.grad, b.grad: ", w1.grad, b.grad)

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
# 输入数量为二, 对应X_train每个示例有两个特征
# 输出数量为二, 对应Y_train有两个类别(0,1)
model = NeuralNetwork(2, 2)

# 获得一个device类, 可以传给model.to, 指定设备类型(如gpu). 也可以直接传给它'cuda'之类的字符串
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)  # 兼容写法. mps中pytorch有些操作受限制, 遇到了可以临时设置函数传参指定用cpu计算
model = model.to(device)

# 访问权重参数矩阵 model.layers[0].weight
print("weight shape: ", model.layers[0].weight.shape)

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

# print('model.parameters()', model.parameters())
# 全连接层中, 每一层每个神经元都与上一层(若有)每个神经元有一条连接, 每个连接对应一个权重, 然后每个神经元都有一个偏置
# 两者数量加起来就是模型全部参数的数量
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
num_epochs = 3

# 对模型进行多轮次的训练
for epoch in range(num_epochs):
    model.train()  # 模型设为训练模式
    for batch_idx, (features, labels) in enumerate(train_loader):
        features, labels = features.to(device), labels.to(device)
        logits = model(features)

        loss = F.cross_entropy(logits, labels)

        # 置零梯度
        optimizer.zero_grad()
        # 计算梯度
        loss.backward()
        # 利用梯度更新模型参数, 以最小化损失
        optimizer.step()

        # logging
        print(
            f"Epoch: {epoch + 1:03d}/{num_epochs:03d}"
            f" | Batch {batch_idx}/{len(train_loader):03d}"
            f" | Train Loss: {loss:.2f}"
        )

    # 模型设为计算模式
    model.eval()
    # 用模型进行计算
    with torch.no_grad():  # 只向前传播, 不计算梯度
        outputs = model(
            X_train.to(device)
        )  # 对训练数据集进行计算, 获得一个(5, 2)的张量作为计算结果
    print("outputs: ", outputs)
    # 设置打印格式, 关掉科学计数法格式, 以普通小数显示
    torch.set_printoptions(sci_mode=False)
    # 为获取类别成员概率, 使用softmax函数, 将outputs转成概率分布. 在dim=1维度(dim从0起), 即类别维度, 进行归一化, 得到每个样本属于类别0和类别1的概率
    # 也可以设dim=-1, 对最后一个维度(一般都对应于类别)进行归一化
    probas = torch.softmax(outputs, dim=1)
    print("probas: ", probas)
    # 输出: tensor([[0.9938, 0.0062],
    #     [0.9903, 0.0097],
    #     [0.9810, 0.0190],
    #     [0.0171, 0.9829],
    #     [0.0075, 0.9925]])
    # 代表五个训练示例分别属于标签0和1的概率

    # 还可以用argmax将这些概率值转为类别标签预测:
    predictions = torch.argmax(probas, dim=1)
    print("predictions: ", predictions)

    compare = predictions == Y_train.to(device)

    # 与真实的训练标签做比较, 预期打印 tensor([True, True, True, True, True])
    print("compare: ", compare)

    # 计算预测正确的数量, 预期为5
    torch.sum(compare)


# 封装一个函数, 使预测准确率的计算更加通用
def compute_accuracy(model, dataloader):
    # 获取模型当前所在的设备
    device = next(model.parameters()).device
    model = model.eval()
    correct = 0.0
    total_example = 0

    for idx, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)
        with torch.no_grad():
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_example += len(compare)

    return (
        correct / total_example
    ).item()  # 调用item会将张量的值以python浮点数形式返回

# 将函数用于训练数据
print(compute_accuracy(model, train_loader))  # 预期返回1.0
# 将函数用于测试数据
print(compute_accuracy(model, test_loader))  # 预期返回1.0

# 保存训练好的模型到硬盘中: model.state_dict返回一个字典对象, 可将模型中每一层映射到其可训练的参数(权重和偏置). model.pth是保存的文件名, 一般用.pth或.pt作为后缀
# model.state_dict获得的键名大概是这样的:
# ['layers.0.weight', 'layers.0.bias', 'layers.2.weight', 'layers.2.bias', 'layers.4.weight', 'layers.4.bias']
torch.save(model.state_dict(), "model.pth")
print(model.state_dict().keys())

# 从硬盘读取保存的模型:
model = NeuralNetwork(2, 2)  # 这行不是必需的
model = model.to(device)
# torch.load 读取文件, 重建上述字典对象. load_state_dict将该对象中的参数应用到模型中
model.load_state_dict(torch.load("model.pth"))

# 测试是否可使用gpu (包含 cuda 与 mps)
print("CUDA is available: ", torch.cuda.is_available())
print("MPS is available: ", torch.backends.mps.is_available())
print("Using device: ", device)

# 在当前可用设备上计算张量
tensor_1 = torch.tensor([1.0, 2.0, 3.0])
tensor_2 = torch.tensor([4.0, 5.0, 6.0])
# 将张量转移到当前最合适的设备上并执行相关操作
tensor_1 = tensor_1.to(device)
tensor_2 = tensor_2.to(device)
print(
    tensor_1 + tensor_2
)  # 参与计算的张量必须处于同一个设备上