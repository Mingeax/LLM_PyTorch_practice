import torch
from torch import nn

from ..gpt_dataset import inputsLen
from ..gpt_dataset import sample_embeddings as inputs

# 计算单个输入向量x_2的注意力分数点积
query = inputs[1]
attn_scores_2 = torch.empty(inputsLen)
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print("attn_scores_2: ", attn_scores_2)

# 归一化, 获得注意力权重
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention Weights: ", attn_weights_2_tmp)
print("attn_weights_2_tmp.sum(): ", attn_weights_2_tmp.sum())


# 或者, 使用softmax归一化, 手动实现:
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)


attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Attention Weights: ", attn_weights_2_tmp)
print("attn_weights_2_tmp.sum():", attn_weights_2_tmp.sum())

# 更建议使用pytorch封装的实现:
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention Weights: ", attn_weights_2_tmp)
print("attn_weights_2_tmp.sum():", attn_weights_2_tmp.sum())


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
print("attn_scores1: ", attn_scores)

# 或者用矩阵乘法计算
attn_scores = inputs @ inputs.T
print("attn_scores2: ", attn_scores)

# 归一化
attn_weights = torch.softmax(attn_scores, dim=-1)
print("attn_weights: ", attn_weights)

# 验证每一行总和是否为1
print("All row sums: ", attn_weights.sum(dim=-1))
# 计算所有上下文向量
all_context_vecs = attn_weights @ inputs  # 矩阵相乘
print("all_context_vecs: ", all_context_vecs)

# 实现可训练权重, 即缩放点积注意力
x_2 = inputs[1]  # 第二个输入元素
d_in = inputs.shape[1]  # 输入的嵌入维度为3
d_out = 2  # 输出的嵌入维度为2

torch.manual_seed(123)
# 初始化三个权重矩阵, 用于生成查询向量, 值向量和键向量
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# 第二个输入词元的查询向量, 键向量和值向量
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_key
print("query_2: ", query_2)

# 用矩阵乘法可以得到所有的键向量和值向量
keys = inputs @ W_key
values = inputs @ W_value
print("keys.shape: ", keys.shape)
print("values.shape: ", values.shape)

keys_2 = keys[1]  # 等于 x_2 @ W_key
# 计算第二个输入词元对自身的注意力分数
attn_score_22 = query_2.dot(keys_2)
print("attn_score_22: ", attn_score_22)

# 计算第二个输入词元对所有输入词元的注意力分数
attn_scores_2 = query_2 @ keys.T
print("attn_scores_2: ", attn_scores_2)

# 将注意力分数缩放(除以键向量嵌入维度的平方根), 并应用softmax归一化得到注意力权重
# 除以维度平方根是为了减小点积, 增加反向传播时的梯度
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print("attn_weights_2: ", attn_weights_2)

# 对值向量进行加权求和, 得到第二个输入词元的上下文向量
# 把注意力权重当作加权因子
context_vec_2 = attn_scores_2 @ values
print("attn_scores_2: ", attn_scores_2)


# 实现一个简化的自注意类
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
        self.W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
        self.W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

    def forward(self, x):
        queries = x @ self.W_query
        keys = x @ self.W_key
        values = x @ self.W_value
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec


# 使用方法
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print("sa_v1(inputs): ", sa_v1(inputs))


# 使用nn.Linear层进一步优化SelfAttention_v1的实现
class SelfAttention_v2(nn.Module):
    def init(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        # nn.Linear提供了优化过的权重初始方案
        self.W_query = nn.Linear(torch.rand(d_in, d_out), requires_grad=False)
        self.W_key = nn.Linear(torch.rand(d_in, d_out), requires_grad=False)
        self.W_value = nn.Linear(torch.rand(d_in, d_out), requires_grad=False)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec


# 使用方法同SelfAttention_v1
torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))

# 因果注意力(掩码注意力)
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(input)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

print("attn_weights:", attn_weights)

context_length = attn_scores.shape[0]
# 获得下三角方阵, 元素为1和0
mask_simple = torch.tril(torch.ones(context_length, context_length))
print("mask_simple: ", mask_simple)

masked_simple = attn_weights * mask_simple
print("masked_simple: ", masked_simple)

# 重新归一化注意力权重
row_nums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_nums
print("masked_simple_norm: ", masked_simple_norm)

# 或者使用优化的掩码方式: 获得上三角方阵, 元素为1和0
mask = torch.triu(
    torch.ones(context_length, context_length), diagonal=1
)  # diagonal是矩阵三角偏移量
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)  # 上三角设为负无穷
print("masked: ", masked)

attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=1)
print("attn_weights: ", attn_weights)

# dropout
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)  # 50%的dropout率, 即随机丢弃矩阵中一半的元素
example = torch.ones(6, 6)
print("dropout(example):", dropout(example))


# 因果注意力和dropout的注意力类(简化的)
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)  # 多一个dropout层
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec


# 使用方法:
# torch.manual_seed(123)
# context_length = batch.shape[1]
# ca = CausalAttention(d_in, d_out, context_length, 0.0)
# context_vecs = ca(batch)
# print("context_vecs.shape: ", context_vecs.shape)  # 三维向量 ([2,6,2])
