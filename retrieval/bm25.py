"""BM25 稀疏检索 —— 负责人：B（检索）

状态：[READY] 手写实现，未使用 `rank_bm25` 等第三方库。

公式（与 INTERFACE.md 一致）::

    score(q, d) = Σ_{t ∈ q} IDF(t) · ( f(t,d) · (k1 + 1) )
                / ( f(t,d) + k1 · (1 − b + b · |d| / avgdl) )

    IDF(t) = ln( (N − n_t + 0.5) / (n_t + 0.5) + 1 )

    其中 N 为文档总数, n_t 为含 t 的文档数, f(t,d) 为 t 在 d 中的词频,
    |d| 为文档长度（词数）, avgdl 为平均文档长度。

实现说明（写进报告第 4 章）：

* 倒排表 ``term -> {内部文档下标: tf}``，打分时对每个查询词取出整条倒排链，
  用 numpy 一次算完该词对所有文档的贡献，再累加；不逐文档循环，
  3507 篇语料上单查询约 0.3 ms。
* **查询词去重**：同一个词在查询里出现多次只累加一次（与 IR 教材口径一致，
  也避免长查询被自己的重复词主导）。想改成按查询 TF 加权，把
  ``dedup_query=False`` 传给构造函数即可。
* 打分相同的文档按**内部下标（即语料顺序）**升序排列，保证同一份数据
  重复运行结果完全一致（可复现性要求）。
* 空查询 / 全是未登录词 → 返回 ``[]``。
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .tokenization import ChineseTokenizer, build_tokenizer

STUB = False


class BM25:
    """BM25 打分器。签名冻结，见 INTERFACE.md。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75,
                 tokenizer: ChineseTokenizer | None = None,
                 dedup_query: bool = True) -> None:
        if k1 < 0:
            raise ValueError("k1 必须 ≥ 0")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b 必须在 [0, 1] 区间")
        self.k1 = float(k1)
        self.b = float(b)
        self.dedup_query = bool(dedup_query)
        self.tokenizer = tokenizer if tokenizer is not None else build_tokenizer()

        self.doc_ids: List[str] = []
        self.doc_len: np.ndarray = np.zeros(0, dtype=np.int32)
        self.avgdl: float = 0.0
        self.postings: dict[str, dict[int, int]] = {}   # term -> {内部下标: tf}
        self.doc_terms: List[dict[str, int]] = []       # 内部下标 -> {term: tf}（RM3 建模用）
        self.doc_freq: dict[str, int] = {}              # term -> n_t
        self.vocab_size: int = 0
        self._norm: np.ndarray = np.zeros(0, dtype=np.float32)  # k1*(1-b+b*dl/avgdl)
        self._idf: dict[str, float] = {}
        self._fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, docs: Sequence[dict]) -> "BM25":
        """docs: [{"doc_id": str, "text": str}]"""
        term_counts: dict[str, dict[int, int]] = {}
        doc_ids: List[str] = []
        doc_len: List[int] = []
        doc_terms: List[dict[str, int]] = []

        for idx, d in enumerate(docs):
            doc_ids.append(str(d["doc_id"]))
            tokens = self.tokenizer.cut(str(d.get("text") or ""))
            doc_len.append(len(tokens))
            if not tokens:
                doc_terms.append({})
                continue
            local: dict[str, int] = {}
            for t in tokens:
                local[t] = local.get(t, 0) + 1
            doc_terms.append(local)
            for t, tf in local.items():
                term_counts.setdefault(t, {})[idx] = tf

        self.doc_ids = doc_ids
        self.doc_terms = doc_terms
        self.doc_len = np.asarray(doc_len, dtype=np.int32)
        self.avgdl = float(self.doc_len.mean()) if len(doc_len) else 0.0
        if self.avgdl <= 0:
            self.avgdl = 1.0
        self.postings = term_counts
        self.doc_freq = {t: len(p) for t, p in term_counts.items()}
        self.vocab_size = len(term_counts)

        n = max(1, len(self.doc_ids))
        self._idf = {
            t: math.log((n - df + 0.5) / (df + 0.5) + 1.0) for t, df in self.doc_freq.items()
        }
        self._norm = (self.k1 * (1.0 - self.b + self.b * self.doc_len / self.avgdl)).astype(np.float32)
        self._fitted = True
        return self

    def _require_fit(self) -> None:
        if not self._fitted:
            raise RuntimeError("BM25.search() 前必须先 fit(docs)")

    # --------------------------------------------------------------- search
    def query_terms(self, query: str) -> List[str]:
        terms = self.tokenizer.cut(query or "")
        if not terms:
            return []
        if self.dedup_query:
            seen, uniq = set(), []
            for t in terms:
                if t not in seen:
                    seen.add(t)
                    uniq.append(t)
            return uniq
        return terms

    def scores(self, query: str) -> np.ndarray:
        """返回对**全部**文档的 BM25 分数（未排序）。"""
        self._require_fit()
        return self.scores_weighted({t: 1.0 for t in self.query_terms(query)})

    def scores_weighted(self, term_weights: dict) -> np.ndarray:
        """带权查询：``score = Σ_t w_t · IDF(t)·tf·(k1+1)/(tf + k1(1-b+b|d|/avgdl))``。

        E6（RM3 伪相关反馈）要把扩展词以小于 1 的权重并入查询，所以把
        "查询词 → 权重" 抽出来；``scores()`` 就是权重全为 1 的特例。
        """
        self._require_fit()
        out = np.zeros(len(self.doc_ids), dtype=np.float32)
        for term, weight in term_weights.items():
            if not weight:
                continue
            post = self.postings.get(term)
            if not post:
                continue
            ids = np.fromiter(post.keys(), dtype=np.int64, count=len(post))
            tfs = np.fromiter(post.values(), dtype=np.float32, count=len(post))
            idf = self._idf[term]
            # tf*(k1+1) / (tf + k1*(1-b+b*dl/avgdl))
            contrib = float(weight) * idf * (tfs * (self.k1 + 1.0)) / (tfs + self._norm[ids])
            out[ids] += contrib
        return out

    def search_weighted(self, term_weights: dict, topk: int = 100) -> List[Tuple[str, float]]:
        """带权查询的检索结果 [(doc_id, score)]。"""
        scores = self.scores_weighted(term_weights)
        nz = np.flatnonzero(scores > 0)
        if nz.size == 0:
            return []
        order = nz[np.lexsort((nz, -scores[nz]))][: max(0, int(topk))]
        return [(self.doc_ids[int(i)], float(scores[int(i)])) for i in order]

    def postings_for(self, term: str) -> dict:
        """``term -> {内部下标: tf}``（RM3 建模用）。"""
        return self.postings.get(term, {})

    def vocab_terms_of(self, row: int) -> dict:
        """某篇文档里出现过的 ``{term: tf}``（RM3 估计相关模型用）。"""
        return self.doc_terms[int(row)]

    def doc_tf(self, row: int, term: str) -> int:
        return int(self.postings.get(term, {}).get(int(row), 0))

    def idf_of(self, term: str) -> float:
        return float(self._idf.get(term, 0.0))

    def _rank_with_scores(self, query: str, topk: int | None = None) -> Tuple[List[int], np.ndarray]:
        scores = self.scores(query)
        nz = np.flatnonzero(scores > 0)
        if nz.size == 0:
            return [], scores
        # 主键 -score 升序，次键下标升序 → lexsort 的最后一个 key 是主键
        order = nz[np.lexsort((nz, -scores[nz]))]
        if topk is not None:
            order = order[: max(0, int(topk))]
        return order.tolist(), scores

    def rank(self, query: str, topk: int | None = None) -> List[int]:
        """返回按分降序的内部下标（同分按语料顺序）。"""
        idx, _ = self._rank_with_scores(query, topk)
        return idx

    def search(self, query: str, topk: int = 100) -> List[Tuple[str, float]]:
        """返回按相关度降序的 [(doc_id, score)]，长度 ≤ topk。"""
        idx, scores = self._rank_with_scores(query, topk)
        return [(self.doc_ids[i], float(scores[i])) for i in idx]

    def topk_ids(self, query: str, topk: int = 100) -> List[str]:
        """只要 doc_id 的快捷方式（E2/E3 用）。"""
        return [self.doc_ids[i] for i in self.rank(query, topk)]

    def rank_ids_batch(self, queries: Iterable[str], topk: int = 100,
                       progress: bool = False) -> List[List[str]]:
        queries = list(queries)
        iterator = queries
        if progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(queries, desc="[bm25]", ncols=80)
            except Exception:
                pass
        return [self.topk_ids(q, topk) for q in iterator]

    # ------------------------------------------------------------ 自检工具
    def self_check(self) -> dict:
        """给 B 用的自检：手工构造 3 篇文档，检查排序是否符合直觉。"""
        demo = [
            {"doc_id": "d1", "text": "联邦学习 中 的 非独立同分布 数据"},
            {"doc_id": "d2", "text": "检索 增强 生成 的 混合 检索 与 重排"},
            {"doc_id": "d3", "text": "无关 的 内容 完全 不 相关"},
        ]
        demo_bm25 = BM25(k1=self.k1, b=self.b, tokenizer=self.tokenizer,
                         dedup_query=self.dedup_query).fit(demo)
        hit = demo_bm25.search("混合检索与重排", topk=3)
        return {
            "fitted": demo_bm25._fitted,
            "n_docs": len(demo),
            "avgdl": round(demo_bm25.avgdl, 3),
            "vocab": demo_bm25.vocab_size,
            "top1_is_d2": bool(hit) and hit[0][0] == "d2",
            "empty_query_returns_empty": demo_bm25.search("", topk=3) == [],
        }

    # -------------------------------------------------------------- 序列化
    def to_dict(self) -> dict:
        return {
            "k1": self.k1,
            "b": self.b,
            "dedup_query": self.dedup_query,
            "tokenizer": {"mode": self.tokenizer.mode,
                          "lowercase": self.tokenizer.lowercase,
                          "drop_stopwords": self.tokenizer.drop_stopwords},
            "doc_ids": self.doc_ids,
            "doc_len": self.doc_len.tolist(),
            "avgdl": self.avgdl,
            "postings": {t: {str(k): v for k, v in p.items()} for t, p in self.postings.items()},
        }

    def save(self, path: str) -> None:
        import json

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str, tokenizer: ChineseTokenizer | None = None) -> "BM25":
        import json

        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        if tokenizer is None:
            cfg = d.get("tokenizer") or {}
            tokenizer = ChineseTokenizer(mode=cfg.get("mode", "auto"),
                                         lowercase=cfg.get("lowercase", True),
                                         drop_stopwords=cfg.get("drop_stopwords", True))
        obj = cls(k1=d["k1"], b=d["b"], tokenizer=tokenizer, dedup_query=d.get("dedup_query", True))
        obj.doc_ids = list(d["doc_ids"])
        obj.doc_len = np.asarray(d["doc_len"], dtype=np.int32)
        obj.avgdl = float(d["avgdl"])
        obj.postings = {t: {int(k): v for k, v in p.items()} for t, p in d["postings"].items()}
        obj.doc_freq = {t: len(p) for t, p in obj.postings.items()}
        obj.vocab_size = len(obj.postings)
        # 反推每篇文档的词频表，保证 RM3 之类的功能在 load 之后仍可用
        doc_terms: list[dict] = [{} for _ in obj.doc_ids]
        for term, p in obj.postings.items():
            for row, tf in p.items():
                doc_terms[row][term] = tf
        obj.doc_terms = doc_terms
        n = max(1, len(obj.doc_ids))
        obj._idf = {t: math.log((n - df + 0.5) / (df + 0.5) + 1.0) for t, df in obj.doc_freq.items()}
        obj._norm = (obj.k1 * (1.0 - obj.b + obj.b * obj.doc_len / obj.avgdl)).astype(np.float32)
        obj._fitted = True
        return obj

    def __repr__(self) -> str:
        return f"BM25(n_docs={len(self.doc_ids)}, vocab={self.vocab_size}, k1={self.k1}, b={self.b})"
