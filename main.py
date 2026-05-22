import torch

tensor0d = torch.tensor(1)
tensor1d = torch.tensor([1, 2, 3])
tensor2d = torch.tensor([[1, 2], [3, 4]])
tensor3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(tensor0d.dtype) # 数字的类型, 现为 torch.float32

floatvec = tensor1d.to(torch.float32)