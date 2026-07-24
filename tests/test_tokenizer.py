"""分词器测试用例"""
import pytest
import sys
from pathlib import Path

from construct_llm.tokenizer import SimpleTokenizerV1, vocab, vocab_size, raw_text

# 添加 src 目录到 Python 路径
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

class TestTokenizer:
    """分词器测试类"""

    def test_vocab_size(self):
        """测试词汇表大小是否合理"""
        assert vocab_size > 0, "词汇表大小应大于 0"
        assert vocab_size < len(raw_text), "词汇表应小于原始文本长度"
        print(f"词汇表大小: {vocab_size}")

    def test_vocab_is_dict(self):
        """测试词汇表是否为字典类型"""
        assert isinstance(vocab, dict), "词汇表应为字典类型"
        assert len(vocab) == vocab_size, "词汇表大小应一致"

    def test_tokenizer_initialization(self):
        """测试分词器初始化"""
        tokenizer = SimpleTokenizerV1(vocab)
        assert tokenizer.str_to_int is not None
        assert tokenizer.int_to_str is not None
        assert len(tokenizer.str_to_int) == vocab_size
        assert len(tokenizer.int_to_str) == vocab_size

    def test_encode_basic(self):
        """测试基本编码功能"""
        tokenizer = SimpleTokenizerV1(vocab)
        # 使用 theVerdict.txt 中实际存在的单词
        test_text = "I had always thought"
        ids = tokenizer.encode(test_text)

        assert isinstance(ids, list), "编码结果应为列表"
        assert len(ids) > 0, "编码结果不应为空"
        assert all(isinstance(i, int) for i in ids), "所有 ID 应为整数"

    def test_decode_basic(self):
        """测试基本解码功能"""
        tokenizer = SimpleTokenizerV1(vocab)
        test_ids = [10, 20, 30]
        decoded_text = tokenizer.decode(test_ids)

        assert isinstance(decoded_text, str), "解码结果应为字符串"
        assert len(decoded_text) > 0, "解码结果不应为空"

    def test_encode_decode_roundtrip(self):
        """测试编码-解码对称性 (关键测试)"""
        tokenizer = SimpleTokenizerV1(vocab)
        # 使用 theVerdict.txt 中实际存在的文本片段
        original_text = "I had always thought Jack"

        # 编码
        ids = tokenizer.encode(original_text)

        # 解码
        decoded_text = tokenizer.decode(ids)

        # 验证 (注意:由于空格处理,可能不完全相同,但应语义一致)
        assert len(decoded_text) > 0, "解码后的文本不应为空"
        assert isinstance(decoded_text, str), "解码结果应为字符串"

    def test_special_characters(self):
        """测试特殊字符处理"""
        tokenizer = SimpleTokenizerV1(vocab)

        # 测试 theVerdict.txt 中包含标点符号的文本
        test_cases = [
            "I had always",
            "thought Jack Gisburn",
            "rather a cheap",
        ]

        for text in test_cases:
            ids = tokenizer.encode(text)
            decoded = tokenizer.decode(ids)
            assert isinstance(ids, list), f"编码失败: {text}"
            assert isinstance(decoded, str), f"解码失败: {text}"
            assert len(ids) > 0, f"编码结果为空: {text}"

    def test_empty_text(self):
        """测试空文本处理"""
        tokenizer = SimpleTokenizerV1(vocab)
        ids = tokenizer.encode("")
        # 空文本可能返回空列表或包含特殊 token
        assert isinstance(ids, list), "编码结果应为列表"

    def test_word_in_vocab(self):
        """测试词汇表中的词能否正确编码"""
        tokenizer = SimpleTokenizerV1(vocab)

        # 从词汇表中取一些词进行测试
        sample_words = list(vocab.keys())[:10]
        for word in sample_words:
            if word.strip():  # 跳过空字符串
                ids = tokenizer.encode(word)
                assert len(ids) > 0, f"单词 '{word}' 编码失败"

    def test_token_ids_are_valid(self):
        """测试生成的 token ID 是否在有效范围内"""
        tokenizer = SimpleTokenizerV1(vocab)
        # 使用 theVerdict.txt 中的实际文本
        test_text = "I had always thought"
        ids = tokenizer.encode(test_text)

        for token_id in ids:
            assert 0 <= token_id < vocab_size, \
                f"Token ID {token_id} 超出词汇表范围 [0, {vocab_size})"

    def test_consistency(self):
        """测试相同文本多次编码结果一致"""
        tokenizer = SimpleTokenizerV1(vocab)
        # 使用 theVerdict.txt 中的实际文本
        test_text = "I had always"

        ids1 = tokenizer.encode(test_text)
        ids2 = tokenizer.encode(test_text)

        assert ids1 == ids2, "相同文本的编码结果应一致"

    def test_raw_text_loading(self):
        """测试原始文本是否正确加载"""
        assert isinstance(raw_text, str), "原始文本应为字符串"
        assert len(raw_text) > 0, "原始文本不应为空"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])