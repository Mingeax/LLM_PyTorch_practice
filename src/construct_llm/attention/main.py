import torch

from ..gpt_dataset import inputs, inputsLen

print("🌞 -- main.py:5 -- inputs:", inputs)

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

# 实现可训练权重
x_2 = inputs[1]  # 第二个输入元素
d_in = inputs.shape[1]  # 输入的嵌入维度为3
d_out = 2  # 输出的嵌入维度为2

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_key

print("query_2: ", query_2)

keys = inputs @ W_key
values = inputs @ W_value
print("keys.shape: ", keys.shape)
print("values.shape: ", values.shape)

# TODO: 使用嵌入向量输入