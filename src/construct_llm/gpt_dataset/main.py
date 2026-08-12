import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset

from ..text_data import the_verdict as raw_text


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt,
    batch_size=4,  # 批次大小batch_size是一个超参数, 较小的值可以减少训练时内存占用, 但产生更多噪声
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader


dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
)
data_iter = iter(dataloader)  # 转换为python迭代器, 以通过next()获取下一项
first_batch = next(
    data_iter
)  # 包含两个张量, 分别存储输入词元id和目标词元id, 它们的长度为max_length. 实际大模型训练时不低于256
second_batch = next(data_iter)  # 相比first_batch, 整体左移一个stride
print("first_batch: ", first_batch)

# 尝试不同的批次大小和步幅
# 步幅等于词元序列长度max_length时, 可以避免批次之间重叠, 降低模型过拟合风险
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print("Inputs:\n", inputs)
print("\nTargets:\n", targets)

# 实现绝对位置嵌入
vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)  # 词元ID张量为8*4, 即batch_size*max_length
print("\nInputs shape:\n", inputs.shape)

token_embeddings = token_embedding_layer(inputs)
print("token_embeddings: ", token_embeddings)

# 创建一个维度与token_embedding_layer相同的嵌入层, 以实现绝对位置嵌入
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print('pos_embeddings: ', pos_embeddings)
