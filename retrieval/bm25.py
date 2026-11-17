"""BM25 稀疏检索 —— 负责人：B（检索）

状态：[STUB] 空实现，等 B 替换为真实现。
本文件必须**手写**，不许 `pip install rank_bm25`：
报告第 4 章「核心算法的实现过程」与「自实现的三处」其中之一就是它。

公式（写进报告）::

    score(q, d) = Σ_{t ∈ q} IDF(t) · ( f(t,d) · (k1 + 1) )
                / ( f(t,d) + k1 · (1 − b + b · |d| / avgdl) )

    IDF(t) = ln( (N − n_t + 0.5) / (n_t + 0.5) + 1 )

    其中 N 为文档总数, n_t 为含 t 的文档数, f(t,d) 为 t 在 d 中的词频,
    |d| 为文档长度（词数）, avgdl 为平均文档长度。

实现清单（B 逐条打勾）：
  [ ] __init__ 里 fit 之前不许抛异常
  [ ] fit(): jieba 分词 → 词表倒排 → 文档长度 → 平均长度
  [ ] IDF 用上面的平滑式，注意 +1 在括号内
  [ ] 查询词去重（同一查询词出现多次只计一次，或按 TF 计，二选一并在报告里说明）
  [ ] search(): 返回按分降序的 [(doc_id, score)]，长度 ≤ topk
  [ ] 空查询 / 未登录词 → 返回 []
  [ ] 与 rrf.fuse() 联调：score 数量级不必和别人一致，RRF 只看排名
"""

from __future__ import annotations

import warnings

STUB = True
_WARNED = False


def _warn_once() -> None:
    global _WARNED
    if not _WARNED:
        warnings.warn(
            "retrieval/bm25.py 仍是 STUB，返回空结果；B 请替换为真实现。",
            RuntimeWarning,
            stacklevel=2,
        )
        _WARNED = True


class BM25:
    """BM25 打分器。签名冻结，见 INTERFACE.md。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.doc_len: list[int] = []
        self.avgdl: float = 0.0
        self.postings: dict[str, dict[str, int]] = {}   # term -> {doc_id: tf}
        self.doc_freq: dict[str, int] = {}              # term -> n_t
        self._fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, docs: list[dict]) -> None:
        """docs: [{"doc_id": str, "text": str}]"""
        _warn_once()
        # TODO(B): 实现分词、倒排、文档长度与 avgdl
        # import jieba
        # for d in docs:
        #     toks = jieba.lcut(d["text"])
        #     ...
        self.doc_ids = [d["doc_id"] for d in docs]
        self._fitted = True

    # --------------------------------------------------------------- search
    def search(self, query: str, topk: int = 100) -> list[tuple[str, float]]:
        """返回按相关度降序的 [(doc_id, score)]，长度 ≤ topk。"""
        _warn_once()
        if not self._fitted:
            raise RuntimeError("BM25.search() 前必须先 fit(docs)")
        # TODO(B): 打分 + 排序 + 截断
        return []

    # ------------------------------------------------------------ 自检工具
    def self_check(self) -> dict:
        """给 B 用的自检：手工构造 3 篇文档，检查排序是否符合直觉。"""
        _warn_once()
        demo = [
            {"doc_id": "d1", "text": "联邦学习 中 的 非独立同分布 数据"},
            {"doc_id": "d2", "text": "检索 增强 生成 的 混合 检索 与 重排"},
            {"doc_id": "d3", "text": "无关 的 内容 完全 不 相关"},
        ]
        self.fit(demo)
        return {"fitted": self._fitted, "n_docs": len(self.doc_ids),
                "avgdl": self.avgdl, "n_terms": len(self.postings)}