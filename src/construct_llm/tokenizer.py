import re
from pathlib import Path

script_dir = Path(__file__).resolve().parent  # __file__ 代表当前脚本文件的路径

file_path = script_dir / "assets" / "theVerdict.txt"  # 相对路径
raw_text = file_path.read_text(encoding="utf-8")

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print("vocab size:", vocab_size)

# 构建词汇表
vocab = {token: integer for integer, token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print("vocab item:", item)
    if i >= 50:
        break


# 一个简单的分词器类
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    # 依照词汇表, 把完整文本转换为词元id列表
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]

        return ids

    # 依照逆向词汇表, 把词元id列表转换为完整文本
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(
            r'\s+([,.?!"()\'])', r"\1", text
        )  # \1代表第一个捕获组, 相当于js里的$1

        return text


# 用分词器类获取文本对应的词元id列表, 并逆向映射回文本
tokenizer = SimpleTokenizerV1(vocab)

ids = tokenizer.encode(raw_text)
print("ids: ", ids)
print("decode ids: ", tokenizer.decode(ids))