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
inputsLen = inputs.shape[0]

print("Inputs:\n", inputs)
print("\nTargets:\n", targets)

# 实现绝对位置嵌入
vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# 每个文本样本有四个词元
max_length = 4
# 每个数据批次包含八个文本样本
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
# torch.arange 创建一个包含在指定半开区间 [start, end) 内按照步长 step 均匀分布的数值的一维张量（1-D Tensor）
# 将此向量作为位置向量
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print("pos_embeddings.shape: ", pos_embeddings.shape)  # 4*256

# 将位置嵌入向量直接加到嵌入向量上
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)  # 8*4*256

print("pos_embeddings: ", pos_embeddings)

# 计算单个输入向量x_2的注意力分数点积
query = inputs[1]
attn_scores_2 = torch.empty(inputsLen)
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print(attn_scores_2)

# 归一化, 获得注意力权重
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention Weights: ", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())


# 或者, 使用softmax归一化, 手动实现:
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)


attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Attention Weights: ", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())

# 更建议使用prytorch封装的实现:
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention Weights: ", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())


# 每个嵌入的输入词元x_i和相应的注意力权重相乘得到向量, 再将其求和, 得到上下文向量, 即所有输入向量的加权总和
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
print(context_vec_2)

# 循环计算所有输入词元的注意力权重
attn_scores = torch.empty(inputsLen, inputsLen)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
print("attn_scores: ", attn_scores)

# 或者用矩阵乘法计算
attn_scores = inputs @ inputs.T
print("attn_scores: ", attn_scores)

# 归一化
attn_weights = torch.softmax(attn_scores, dim=-1)
print("attn_weights: ", attn_weights)

# 验证每一行总和是否为1
print("All row sums: ", attn_weights.sum(dim=-1))

# 计算所有上下文向量
all_context_vecs = attn_weights @ inputs
print("all_context_vecs: ", all_context_vecs)

# TODO: 实现可训练权重
