"""RRF 倒数排名融合 —— 负责人：B（检索）

状态：[STUB] 空实现，等 B 替换为真实现。
必须**手写**（10 行），不许调库：这是报告里「自实现的三处」之二。

公式（写进报告）::

    RRF(d) = Σ_i  w_i / ( k + rank_i(d) )

    rank_i(d) 从 1 开始；d 未出现在第 i 路时，该项贡献 0（不是按缺失排名补）。
    k 为平滑常数（默认 60，Cormack et al. 2009）。
    E2 的 α 设定：稠密路 w = α，BM25 路 w = 1 − α；α ∈ {0,.2,.4,.5,.6,.8,1.0}。

为什么要 RRF：两路分数的量纲完全不同（BM25 是任意实数，稠密是 cos ∈ [-1,1]），
直接加权求和没有物理意义；RRF 只吃排名，天然可比。

实现清单（B 逐条打勾）：
  [ ] runs 是「有序」列表，第 i 个 run 内部已按相关度降序
  [ ] weights 为 None 时等权；长度与 runs 不一致要断言
  [ ] 返回按融合分降序的 [(doc_id, score)]
  [ ] 单路输入时结果应等于该路原始排名（自检用例）
  [ ] 两路完全不重叠时，输出长度 = len(run0) + len(run1)
"""

from __future__ import annotations

import warnings

STUB = True
_WARNED = False


def _warn_once() -> None:
    global _WARNED
    if not _WARNED:
        warnings.warn(
            "retrieval/rrf.py 仍是 STUB，返回空结果；B 请替换为真实现。",
            RuntimeWarning,
            stacklevel=2,
        )
        _WARNED = True


def fuse(runs: list[list[tuple[str, float]]],
         weights: list[float] | None = None,
         k: int = 60) -> list[tuple[str, float]]:
    """倒数排名融合。签名冻结，见 INTERFACE.md。

    Args:
        runs: 多路检索结果，每路是按相关度降序的 [(doc_id, score)]。
        weights: 每路权重；None 表示等权。
        k: RRF 平滑常数，默认 60。

    Returns:
        按融合分降序的 [(doc_id, score)]。
    """
    _warn_once()
    if not runs:
        return []
    if weights is not None:
        assert len(weights) == len(runs), "weights 长度必须与 runs 一致"
    # TODO(B):
    # scores: dict[str, float] = {}
    # for w, run in zip(weights or [1.0] * len(runs), runs):
    #     for rank, (doc_id, _) in enumerate(run, start=1):
    #         scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)
    # return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return []


def self_check() -> dict:
    """给 B 用的自检：单路输入时，排序应与原路完全一致。"""
    _warn_once()
    run = [("a", 9.0), ("b", 5.0), ("c", 1.0)]
    fused = fuse([run])
    return {"input": [d for d, _ in run],
            "fused": [d for d, _ in fused],
            "same": [d for d, _ in run] == [d for d, _ in fused]}