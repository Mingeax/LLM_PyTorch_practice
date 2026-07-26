from importlib.metadata import version

import tiktoken

print("tiktoken version: ", version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2") # 词汇总量有五万多

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terracesof someunknownPlace."
) # someunknownPlace 是一个未知词汇, BPE算法会将其分解为更小的子词单元甚至单个字符, 而无需使用未知词汇词元
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print('integers: ', integers)
