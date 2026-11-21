"""文档切块策略（E3 实验的自变量）。

三种策略：

``whole``
    整段作为一个检索单元（CMRC2018 的金标段落天然就是一段）。
``sentence``
    按句号/问号/感叹号/分号/换行切句，每句一个单元。
``fixed``
    定长滑窗，窗口 ``size`` 字、步长 ``size - overlap``；切点会向最近的
    句末标点回退（``boundary_aware=True``），避免把一句话拦腰截断。

为什么要有 ``boundary_aware``：CMRC 的答案跨度经常正好落在句界上，
硬切会把 2/3 的答案切碎，导致 F1 掉得莫名其妙（这是我们在 D2 踩到的
坑，后来加了句界回退才对齐）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence

_SENT_END = "。！？；!?;\n"
# 句末标点之后允许带上紧跟的引号/括号
_CLOSERS = "”’」』）)】》"


@dataclass(frozen=True)
class ChunkStrategy:
    """切块策略配置。"""

    name: str = "whole"
    size: int = 512
    overlap: int = 0
    boundary_aware: bool = True

    def __post_init__(self) -> None:
        if self.name not in {"whole", "sentence", "fixed"}:
            raise ValueError(f"未知切块策略: {self.name}")
        if self.name == "fixed":
            if self.size <= 0:
                raise ValueError("size 必须为正")
            if not 0 <= self.overlap < self.size:
                raise ValueError("overlap 必须满足 0 <= overlap < size")


@dataclass(frozen=True)
class Chunk:
    """一个检索单元。``doc_id`` 指回原始段落，用于把检索命中映射成文档命中。"""

    chunk_id: int
    doc_id: int
    text: str
    start: int
    end: int


def split_sentences(text: str) -> List[str]:
    """按中文句末标点切句，保留标点。"""
    sents: List[str] = []
    buf: List[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        buf.append(ch)
        if ch in _SENT_END:
            j = i + 1
            while j < len(text) and text[j] in _CLOSERS:
                buf.append(text[j])
                j += 1
            sent = "".join(buf).strip()
            if sent:
                sents.append(sent)
            buf = []
            i = j
            continue
        i += 1
    tail = "".join(buf).strip()
    if tail:
        sents.append(tail)
    return sents


def _fixed_chunks(text: str, size: int, overlap: int, boundary_aware: bool) -> List[tuple[int, int]]:
    """返回 ``[(start, end), ...]``。"""
    n = len(text)
    if n <= size:
        return [(0, n)]
    step = size - overlap
    spans: List[tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(start + size, n)
        if boundary_aware and end < n:
            # 在当前窗口后半段里找最后一个句末标点，找不到就硬切
            window = text[start:end]
            cut = max((window.rfind(c) for c in _SENT_END), default=-1)
            if cut >= size // 2:
                end = start + cut + 1
        spans.append((start, end))
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return spans


def build_chunks(
    docs: Sequence[str],
    strategy: ChunkStrategy | None = None,
) -> List[Chunk]:
    """把 ``docs`` 切成 ``Chunk`` 列表，``chunk_id`` 全局连续。"""
    strategy = strategy or ChunkStrategy()
    chunks: List[Chunk] = []
    for doc_id, doc in enumerate(docs):
        doc = str(doc)
        if not doc.strip():
            continue
        if strategy.name == "whole":
            chunks.append(Chunk(len(chunks), doc_id, doc, 0, len(doc)))
        elif strategy.name == "sentence":
            pos = 0
            for sent in split_sentences(doc):
                start = doc.find(sent, pos)
                if start < 0:
                    start = pos
                end = start + len(sent)
                chunks.append(Chunk(len(chunks), doc_id, sent, start, end))
                pos = end
        else:  # fixed
            for start, end in _fixed_chunks(doc, strategy.size, strategy.overlap, strategy.boundary_aware):
                text = doc[start:end].strip()
                if text:
                    chunks.append(Chunk(len(chunks), doc_id, text, start, end))
    return chunks


def strategy_grid(name: str = "e3") -> List[ChunkStrategy]:
    """E3 的默认策略网格：整段 → 句级 → 128/256/512 定长。"""
    if name != "e3":
        raise ValueError(f"未知网格: {name}")
    return [
        ChunkStrategy("whole"),
        ChunkStrategy("sentence"),
        ChunkStrategy("fixed", size=128, overlap=0),
        ChunkStrategy("fixed", size=256, overlap=0),
        ChunkStrategy("fixed", size=256, overlap=64),
        ChunkStrategy("fixed", size=512, overlap=128),
    ]


def chunks_to_doc_ranking(chunk_ids: Iterable[int], chunks: Sequence[Chunk]) -> List[int]:
    """把 chunk 排名压平成 doc 排名（同文档只保留首次出现，保持顺序）。"""
    seen = set()
    out: List[int] = []
    for cid in chunk_ids:
        doc_id = chunks[cid].doc_id
        if doc_id not in seen:
            seen.add(doc_id)
            out.append(doc_id)
    return out
