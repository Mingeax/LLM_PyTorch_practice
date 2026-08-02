"""文本数据模块 - 包含项目使用的语料库数据"""
from pathlib import Path

# 获取当前脚本所在目录
script_dir = Path(__file__).resolve().parent

# 读取 theVerdict.txt 文件
the_verdict_path = script_dir.parent / "assets" / "theVerdict.txt"
the_verdict = the_verdict_path.read_text(encoding="utf-8")

# 导出语料库数据
__all__ = ["the_verdict"]