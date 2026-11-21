"""B 负责模块的自检（`pytest -q` 一把跑完，不依赖模型与网络）

覆盖：

* BM25：与手算公式逐项对齐、同分按语料顺序、空查询/未登录词、序列化往返；
* RRF：按名次累加、权重、缺失路、k 校验、重复 doc_id；
* 切块：整段/句级/定长+重叠的覆盖性与边界回退；
* 分词：jieba 与 ngram 两条路都给出非空 token；
* 编码器/索引：hash 后端可跑、索引 top-k 与暴力内积一致；
* 评测封装：``eval.retrieval_eval`` 与 ``eval.metrics`` 数值一致。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import metrics as M  # noqa: E402
from eval import retrieval_eval as RE  # noqa: E402
from retrieval.bm25 import BM25  # noqa: E402
from retrieval.chunking import ChunkStrategy, build_chunks, chunks_to_doc_ranking  # noqa: E402
from retrieval.hash_encoder import HashEncoder  # noqa: E402
from retrieval.rrf import fuse  # noqa: E402
from retrieval.tokenization import ChineseTokenizer  # noqa: E402


# ------------------------------------------------------------------ BM25
def _toy_docs():
    return [
        {"doc_id": "d1", "text": "检索 增强 生成"},
        {"doc_id": "d2", "text": "检索 检索 重排"},
        {"doc_id": "d3", "text": "无关 内容"},
    ]


def test_bm25_matches_hand_computation():
    tok = ChineseTokenizer(mode="ngram")
    bm25 = BM25(k1=1.5, b=0.75, tokenizer=tok).fit(_toy_docs())
    # 用切词后的真实 tf/dl 反算单文档公式，逐项核对实现
    term = next(t for t in bm25.query_terms("检索") if t in bm25.postings)
    n_docs = 3
    df = bm25.doc_freq[term]
    idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
    dl = float(bm25.doc_len[0])
    tf = bm25.postings[term][0]
    want = idf * tf * (1.5 + 1.0) / (tf + 1.5 * (1 - 0.75 + 0.75 * dl / bm25.avgdl))
    assert bm25.scores_weighted({term: 1.0})[0] == pytest.approx(want, rel=1e-6)
    # d2 里"检索"出现两次，分数应高于 d1
    assert bm25.scores("检索")[1] > bm25.scores("检索")[0]


def test_bm25_tie_stable_and_empty():
    bm25 = BM25(tokenizer=ChineseTokenizer("ngram")).fit([
        {"doc_id": "a", "text": "苹果"},
        {"doc_id": "b", "text": "苹果"},
    ])
    assert [d for d, _ in bm25.search("苹果", topk=2)] == ["a", "b"]
    assert bm25.search("", topk=5) == []
    assert bm25.search("不存在的词汇xyz", topk=5) == []


def test_bm25_serialization_roundtrip():
    bm25 = BM25(tokenizer=ChineseTokenizer("ngram")).fit(_toy_docs())
    path = ROOT / ".cache" / "tests" / "bm25.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    bm25.save(path)
    back = BM25.load(path)
    assert back.search("检索", topk=2) == bm25.search("检索", topk=2)


def test_bm25_weighted_query_equals_plain_when_unit_weights():
    bm25 = BM25(tokenizer=ChineseTokenizer("ngram")).fit(_toy_docs())
    terms = {t: 1.0 for t in bm25.query_terms("检索")}
    assert np.allclose(bm25.scores("检索"), bm25.scores_weighted(terms))


# ------------------------------------------------------------------ RRF
def test_rrf_basic_and_weights():
    run_a = [("x", 9.0), ("y", 8.0)]
    run_b = [("y", 0.9), ("x", 0.1)]
    fused = dict(fuse([run_a, run_b], k=60))
    assert fused["x"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["y"] == pytest.approx(1 / 62 + 1 / 61)
    weighted = dict(fuse([run_a, run_b], weights=[1.0, 0.0], k=60))
    assert weighted["x"] == pytest.approx(1 / 61)
    assert "y" in weighted  # 权重 0 的路仍有 0 分贡献，但排名靠后


def test_rrf_missing_and_validation():
    a = [("x", 1.0)]
    b = [("y", 1.0)]
    ranked = fuse([a, b], k=60)
    assert {d for d, _ in ranked} == {"x", "y"}, "两路各命中一篇，都应保留"
    with pytest.raises(ValueError):
        fuse([a], k=0)
    with pytest.raises(ValueError):
        fuse([a, b], weights=[1.0])
    # 同一路里重复召回只算最好名次，不重复累加
    dup = dict(fuse([[("x", 1.0), ("x", 2.0)], [("y", 1.0)]], k=60))
    assert dup["x"] == pytest.approx(1 / 61)


# ------------------------------------------------------------------ 切块
def test_chunking_covers_text():
    text = "第一句话。第二句话比较长一些！第三句？" * 3
    for strat in (ChunkStrategy("whole"),
                  ChunkStrategy("sentence"),
                  ChunkStrategy("fixed", size=16, overlap=4),
                  ChunkStrategy("fixed", size=32, overlap=0)):
        chunks = build_chunks([text], strat)
        assert chunks, f"{strat} 没切出块"
        assert all(c.text for c in chunks)
        start = 0
        for c in chunks:  # 顺序覆盖、不重叠倒退
            assert c.start >= start or strat.name != "fixed"
            start = c.start
        assert max(c.end for c in chunks) <= len(text)


def test_chunking_boundary_aware_keeps_sentence():
    text = "一二三四五六七八九十。十一十二十三十四十五。"
    chunks = build_chunks([text], ChunkStrategy("fixed", size=12, overlap=0, boundary_aware=True))
    assert chunks[0].text.endswith("。"), "边界回退应把切口留在句号后"


def test_chunks_to_doc_ranking_dedup():
    chunks = build_chunks(["aa。bb。", "cc。"], ChunkStrategy("sentence"))
    doc_rank = chunks_to_doc_ranking([c.chunk_id for c in chunks], chunks)
    assert doc_rank == list(dict.fromkeys(c.doc_id for c in chunks))


# ------------------------------------------------------------------ 分词
@pytest.mark.parametrize("mode", ["jieba", "ngram"])
def test_tokenizer_modes(mode):
    try:
        tok = ChineseTokenizer(mode=mode)
    except RuntimeError:
        pytest.skip("环境没有 jieba")
    toks = tok.cut("混合检索与重排策略 RAG2024")
    assert toks and all(t.strip() for t in toks)


# ------------------------------------------------------------------ 编码器 / 索引
def test_hash_encoder_and_index_agree_with_numpy():
    from retrieval.bge_encoder import FaissIndex

    enc = HashEncoder(dim=64, ngram=2)
    texts = ["中国的首都是北京", "世界最高峰是珠穆朗玛峰", "苹果公司的产品"]
    emb = enc.encode(texts)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)
    idx = FaissIndex().build([f"d{i}" for i in range(len(texts))], emb)
    q = enc.encode_queries(["首都在哪里"])[0]
    got = idx.search_ids(q, topk=2)
    brute = emb @ q
    want = [f"d{i}" for i in np.argsort(-brute, kind="stable")[:2]]
    assert got == want


# ------------------------------------------------------------------ 评测封装
def test_retrieval_eval_matches_metrics_module():
    rankings = [["a", "b", "c"], ["x", "y"], ["z"]]
    golds = ["c", "y", "q"]
    met = RE.metrics_dict(rankings, golds, ks=(1, 5), ndcg_k=5)
    assert met["recall@1"] == pytest.approx(sum(M.recall_at_k(r, g, 1)
                                                for r, g in zip(rankings, golds)) / 3)
    assert met["recall@5"] == pytest.approx(2 / 3)
    assert met["mrr"] == pytest.approx((1 / 3 + 1 / 2 + 0) / 3)
    lo, hi = RE.recall_ci(rankings, golds, 5, n_boot=200)
    assert lo <= met["recall@5"] <= hi
