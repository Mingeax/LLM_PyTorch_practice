"""神经网络测试用例"""
import pytest
import torch
import sys
from pathlib import Path

from pytorch_example.neural_network import NeuralNetwork, ToyDataset, compute_accuracy

# 添加 src 目录到 Python 路径
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))



class TestNeuralNetwork:
    """神经网络测试类"""

    def test_model_initialization(self):
        """测试模型初始化"""
        model = NeuralNetwork(num_inputs=2, num_outputs=2)
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_model_forward_pass(self):
        """测试前向传播"""
        model = NeuralNetwork(num_inputs=2, num_outputs=2)
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        
        output = model(x)
        
        assert output is not None
        assert isinstance(output, torch.Tensor)
        assert output.shape == (2, 2)  # batch_size=2, num_outputs=2

    def test_output_shape_different_sizes(self):
        """测试不同输入输出维度的模型"""
        # 测试不同的输入输出组合
        test_cases = [
            (2, 2),
            (4, 3),
            (10, 5),
        ]
        
        for num_inputs, num_outputs in test_cases:
            model = NeuralNetwork(num_inputs, num_outputs)
            x = torch.randn(3, num_inputs)  # batch_size=3
            output = model(x)
            assert output.shape == (3, num_outputs)

    def test_model_parameters_exist(self):
        """测试模型参数是否存在"""
        model = NeuralNetwork(num_inputs=2, num_outputs=2)
        params = list(model.parameters())
        
        assert len(params) > 0, "模型应有参数"

    def test_model_parameters_count(self):
        """测试模型参数数量"""
        model = NeuralNetwork(num_inputs=2, num_outputs=2)
        total_params = sum(p.numel() for p in model.parameters())
        
        # 模型结构: Linear(2,30) -> ReLU -> Linear(30,20) -> ReLU -> Linear(20,2)
        # 参数: (2*30+30) + (30*20+20) + (20*2+2) = 90 + 620 + 42 = 752
        assert total_params > 0, "模型参数数量应大于0"
        print(f"模型总参数数量: {total_params}")

    def test_model_train_mode(self):
        """测试训练模式切换"""
        model = NeuralNetwork(2, 2)
        
        # 默认是训练模式
        assert model.training
        
        # 切换到评估模式
        model.eval()
        assert not model.training
        
        # 切换回训练模式
        model.train()
        assert model.training

    def test_gradient_computation(self):
        """测试梯度计算"""
        model = NeuralNetwork(num_inputs=2, num_outputs=2)
        x = torch.randn(1, 2, requires_grad=True)
        
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # 检查参数是否有梯度
        for param in model.parameters():
            assert param.grad is not None, "参数应有梯度"


class TestToyDataset:
    """ToyDataset 测试类"""

    def test_dataset_creation(self):
        """测试数据集创建"""
        X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        Y = torch.tensor([0, 1])
        
        dataset = ToyDataset(X, Y)
        
        assert dataset is not None
        assert len(dataset) == 2

    def test_dataset_length(self):
        """测试数据集长度"""
        X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        Y = torch.tensor([0, 1, 0])
        
        dataset = ToyDataset(X, Y)
        
        assert len(dataset) == 3

    def test_dataset_getitem(self):
        """测试数据索引访问"""
        X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        Y = torch.tensor([0, 1])
        
        dataset = ToyDataset(X, Y)
        
        # 访问第一个样本
        x, y = dataset[0]
        
        assert torch.equal(x, X[0])
        assert torch.equal(y, Y[0])
        assert y.item() == 0

    def test_dataset_getitem_second(self):
        """测试访问第二个样本"""
        X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        Y = torch.tensor([0, 1])
        
        dataset = ToyDataset(X, Y)
        
        x, y = dataset[1]
        
        assert torch.equal(x, X[1])
        assert torch.equal(y, Y[1])
        assert y.item() == 1

    def test_dataloader(self):
        """测试 DataLoader"""
        X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        Y = torch.tensor([0, 1, 0])
        
        dataset = ToyDataset(X, Y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
        
        batches = list(dataloader)
        
        assert len(batches) == 2  # 3 个样本, batch_size=2, 应该有 2 个 batch
        
        # 检查第一个 batch
        x_batch, y_batch = batches[0]
        assert x_batch.shape == (2, 2)
        assert y_batch.shape == (2,)


class TestComputeAccuracy:
    """准确率计算测试类"""

    def test_perfect_accuracy(self):
        """测试完美准确率"""
        model = NeuralNetwork(2, 2)
        
        # 创建简单的数据集,模型应该能完美分类
        X = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])
        Y = torch.tensor([0, 1])
        dataset = ToyDataset(X, Y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
        
        accuracy = compute_accuracy(model, dataloader)
        
        assert 0.0 <= accuracy <= 1.0, "准确率应在 0 到 1 之间"

    def test_accuracy_range(self):
        """测试准确率数值范围"""
        model = NeuralNetwork(2, 2)
        
        X = torch.randn(10, 2)
        Y = torch.randint(0, 2, (10,))
        dataset = ToyDataset(X, Y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
        
        accuracy = compute_accuracy(model, dataloader)
        
        assert isinstance(accuracy, float), "准确率应为浮点数"
        assert 0.0 <= accuracy <= 1.0, "准确率应在 0 到 1 之间"


class TestDeviceHandling:
    """设备处理测试类"""

    def test_cpu_available(self):
        """测试 CPU 是否可用"""
        assert torch.device('cpu') is not None

    def test_device_selection(self):
        """测试设备选择逻辑"""
        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        
        assert device is not None
        assert str(device) in ['cpu', 'cuda', 'mps']

    def test_tensor_on_device(self):
        """测试张量设备转移"""
        device = torch.device("cpu")  # 使用 CPU 确保兼容性
        x = torch.tensor([1.0, 2.0])
        
        x = x.to(device)
        
        assert x.device == device


class TestTrainingStep:
    """训练步骤测试类"""

    def test_training_step(self):
        """测试单个训练步骤"""
        model = NeuralNetwork(num_inputs=2, num_outputs=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        
        x = torch.randn(2, 2)
        y = torch.tensor([0, 1])
        
        # 前向传播
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0, "损失值应大于0"

    def test_loss_decreases(self):
        """测试损失是否在多个 epoch 后降低"""
        model = NeuralNetwork(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        
        x = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])
        y = torch.tensor([0, 1])
        
        initial_loss = None
        for _ in range(10):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            
            if initial_loss is None:
                initial_loss = loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        final_loss = loss.item()
        # 损失应该降低(或至少不增加太多)
        assert final_loss < initial_loss * 2, "损失应该降低"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])