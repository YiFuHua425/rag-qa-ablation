"""中文分词。

BM25 的效果高度依赖分词粒度，因此这里把分词器单独抽出来，保证
建索引与查询用的是同一套切词规则（踩坑记录：早期版本建索引用
``cut_for_search``、查询用 ``cut``，导致查全率莫名掉了 3 个点）。

优先使用 jieba；如果运行环境没有 jieba（例如只装了 numpy 的最小
环境），自动退化为「汉字单字 + 汉字二元组 + 英文单词」的切法，保证
任何机器都能跑通，不至于因为一个分词库卡住实验。
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional

try:  # pragma: no cover - 取决于运行环境
    import jieba

    _HAS_JIEBA = True
except Exception:  # pragma: no cover
    jieba = None  # type: ignore
    _HAS_JIEBA = False


_CJK = r"\u4e00-\u9fff"
_CJK_RUN = re.compile(f"[{_CJK}]+")
_WORD = re.compile(r"[a-zA-Z]+|\d+(?:\.\d+)?")

# 中文高频虚词：BM25 的 idf 会自动压低它们的权重，这里只是少建一些
# 倒排项，降低内存占用，对结果无实质影响。
_STOPWORDS = set(
    "的 了 在 是 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 "
    "自己 这 那 他 她 它 们 与 及 或 而 但 被 把 让 对 从 向 于 为 之 其 等 各 每".split()
)


class ChineseTokenizer:
    """统一的中文分词器。

    Parameters
    ----------
    mode:
        ``"jieba"``（默认）或 ``"ngram"``。``"auto"`` 会在缺少 jieba 时自动降级。
    lowercase:
        是否把英文统一成小写。
    drop_stopwords:
        是否丢弃 ``_STOPWORDS`` 中的虚词。
    """

    def __init__(
        self,
        mode: str = "auto",
        lowercase: bool = True,
        drop_stopwords: bool = True,
    ) -> None:
        if mode == "auto":
            mode = "jieba" if _HAS_JIEBA else "ngram"
        if mode not in {"jieba", "ngram"}:
            raise ValueError(f"未知分词模式: {mode}")
        if mode == "jieba" and not _HAS_JIEBA:
            raise RuntimeError("当前环境没有安装 jieba，请改用 mode='ngram'")
        self.mode = mode
        self.lowercase = lowercase
        self.drop_stopwords = drop_stopwords

    # ------------------------------------------------------------------
    def cut(self, text: str) -> List[str]:
        """把一段文本切成 token 列表（保留顺序，不做去重）。"""
        if not text:
            return []
        text = text.strip()
        if self.lowercase:
            text = text.lower()
        if self.mode == "jieba":
            tokens = [t.strip() for t in jieba.lcut(text)]
        else:
            tokens = self._ngram_cut(text)
        tokens = [t for t in tokens if t and not t.isspace()]
        if self.drop_stopwords:
            tokens = [t for t in tokens if t not in _STOPWORDS]
        return tokens

    def cut_batch(self, texts: Iterable[str]) -> List[List[str]]:
        return [self.cut(t) for t in texts]

    # ------------------------------------------------------------------
    @staticmethod
    def _ngram_cut(text: str) -> List[str]:
        tokens: List[str] = []
        for run in _CJK_RUN.findall(text):
            tokens.extend(run)  # 单字
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))  # 二元组
        tokens.extend(_WORD.findall(text))
        return tokens

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"ChineseTokenizer(mode={self.mode!r})"


def build_tokenizer(mode: Optional[str] = None) -> ChineseTokenizer:
    """按可用性构造分词器；``mode=None`` 表示自动选择。"""
    return ChineseTokenizer(mode or "auto")
