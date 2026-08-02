from importlib.metadata import version

import tiktoken

from ..text_data import the_verdict

print("tiktoken version: ", version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")  # 词汇总量有五万多

text = the_verdict

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print("integers: ", integers)

enc_sample = integers[50:]

context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size]
print(f"x: {x}")
print(f"y:      {y}")

for i in range(1,context_size+1):
  context=enc_sample[:1]
  desired=enc_sample[1]
  # 输入-目标对
  print(tokenizer.decode(context), '---->',tokenizer.decode([desired]))


