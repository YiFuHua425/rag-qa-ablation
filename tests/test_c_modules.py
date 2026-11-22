"""C 模块自检：上下文拼装 / 重排 / 阅读器解码 —— 负责人：C

    python -m pytest tests/test_c_modules.py -q

设计原则：**不下载模型也能跑**。
  * 上下文拼装是纯函数，直接断言；
  * 重排用 ``backend="lexical"`` 的占位后端（只验证排序/缺失候选/截断行为，
    **分数没有意义**，所以断言只看顺序）；
  * 阅读器只测 span 解码（``_best_span``），用合成 logits，不加载权重。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reader import context_builder as CB  # noqa: E402
from reader.extractive_qa import ExtractiveReader  # noqa: E402
from rerank.cross_encoder import Reranker  # noqa: E402

DOCS = ["甲" * 30, "乙" * 30, "丙" * 30, "丁" * 30, "戊" * 30]
IDS = ["a", "b", "c", "d", "e"]


# ---------------------------------------------------------------- 上下文
def test_context_self_check():
    assert CB.self_check()["ok"]


def test_context_order_positions():
    docs = dict(zip(IDS, DOCS))
    for order, checker in (
        ("gold_first", lambda used: used[0] == "c"),
        ("gold_last", lambda used: used[-1] == "c"),
        ("gold_middle", lambda used: used[len(used) // 2] == "c"),
    ):
        _, used = CB.build_context(IDS, docs, order, 1000, gold_doc_id="c")
        assert checker(used), (order, used)
        assert sorted(used) == sorted(IDS), "重排不该丢段落"


def test_context_truncation_and_stats():
    docs = dict(zip(IDS, DOCS))
    _, used = CB.build_context(IDS, docs, "natural", 100)     # 每段 30 字 + 换行
    assert used == ["a", "b", "c"], used
    stats = CB.get_last_stats()
    assert stats["n_used"] == 3 and stats["dropped_doc_ids"] == ["d", "e"]

    _, used = CB.build_context(IDS, docs, "gold_last", 100, gold_doc_id="e")
    stats = CB.get_last_stats()
    assert used == ["a", "b", "c"]
    assert stats["gold_in_candidates"] and not stats["gold_used"], "金标被截断时 gold_used 必须为 False"


def test_context_accepts_dict_map_and_missing_ids():
    docs = {i: {"doc_id": i, "text": t} for i, t in zip(IDS, DOCS)}
    ctx, used = CB.build_context(["zzz"] + IDS[:2], docs, "natural", 1000)
    assert used == ["a", "b"]
    assert CB.get_last_stats()["missing_doc_ids"] == ["zzz"]
    assert ctx == DOCS[0] + "\n" + DOCS[1]


def test_context_rejects_bad_order():
    with pytest.raises(ValueError):
        CB.build_context(IDS, dict(zip(IDS, DOCS)), "gold_side", 100)


# ---------------------------------------------------------------- 重排
@pytest.fixture(scope="module")
def lexical_reranker():
    return Reranker(depth=3, topn=2, device="cpu", backend="lexical",
                    doc_map=dict(zip(IDS, DOCS))).set_doc_map(dict(zip(IDS, DOCS)))


def test_rerank_reorders_within_depth(lexical_reranker):
    reranker = lexical_reranker
    query = "甲甲甲"
    cands = [("b", 0.9), ("c", 0.8), ("a", 0.1), ("d", 0.05)]
    out = reranker.rerank(query, cands, topn=2)
    assert out[0][0] == "a", f"query 与 a 完全重合，应排第一：{out}"
    assert len(out) == 2


def test_rerank_keeps_candidates_beyond_depth(lexical_reranker):
    reranker = lexical_reranker
    cands = [(i, 1.0) for i in IDS]
    out = reranker.rerank("乙", cands, topn=5)
    ids = [d for d, _ in out]
    assert set(ids[:3]) == {"a", "b", "c"}, ids           # depth=3 内被重排
    assert ids[3:] == ["d", "e"], "depth 之外的候选必须原序保留在尾部"


def test_rerank_missing_text_goes_last(lexical_reranker):
    reranker = lexical_reranker
    out = reranker.rerank("甲甲", [("nope", 5.0), ("a", 0.1)], topn=2)
    assert out[-1][0] == "nope" and out[-1][1] == float("-inf")
    assert out[0][0] == "a"


# ---------------------------------------------------------------- 阅读器
def _fake_feature(n_question: int = 4, n_context: int = 6):
    """构造一个 feature：前 n_question 个 token 属问题侧（seq=0），其余属上下文（seq=1）。"""
    seq_ids = [0] * n_question + [1] * n_context
    offsets = [(i, i + 1) for i in range(len(seq_ids))]
    return offsets, seq_ids


def test_reader_picks_context_span_only():
    reader = ExtractiveReader(device="cpu", max_answer_len=10, n_best=5)
    offsets, seq_ids = _fake_feature()
    start = np.full(len(seq_ids), -5.0)
    end = np.full(len(seq_ids), -5.0)
    start[0] = 9.0                                     # 问题侧的最高分：必须被忽略
    start[5] = 6.0
    end[6] = 6.0                                       # 上下文里的 (5,6)
    score, s, e = reader._best_span(start, end, offsets, seq_ids)
    assert (s, e) == (5, 6), (s, e)
    assert score == pytest.approx(12.0)


def test_reader_respects_max_answer_len():
    reader = ExtractiveReader(device="cpu", max_answer_len=2, n_best=5)
    offsets, seq_ids = _fake_feature(n_context=8)
    start = np.full(len(seq_ids), -5.0)
    end = np.full(len(seq_ids), -5.0)
    start[4] = 8.0
    end[9] = 9.0                                       # 跨度 5 > max_answer_len=2
    score, s, e = reader._best_span(start, end, offsets, seq_ids)
    assert e - s < 2, (s, e)


def test_reader_no_valid_span():
    reader = ExtractiveReader(device="cpu")
    offsets, seq_ids = [("", "")] * 3, [-1, -1, -1]     # 全是特殊符号
    start = np.zeros(3)
    end = np.zeros(3)
    score, _s, _e = reader._best_span(start, end, offsets, seq_ids)
    assert not np.isfinite(score)
