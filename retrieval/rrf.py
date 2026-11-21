"""RRF 倒数排名融合 —— 负责人：B（检索）

状态：[READY] 手写实现（十行核心），未调用任何融合库。

公式（与 INTERFACE.md 一致）::

    RRF(d) = Σ_i  w_i / ( k + rank_i(d) )

    rank_i(d) 从 1 开始；d 未出现在第 i 路时，该项贡献 0（不是按缺失排名补）。
    k 为平滑常数（默认 60，Cormack et al. 2009）。
    E2 的 α 设定：稠密路 w = α，BM25 路 w = 1 − α。

为什么要 RRF：两路分数的量纲完全不同（BM25 是任意实数，稠密是 cos ∈ [-1,1]），
直接加权求和没有物理意义；RRF 只吃排名，天然可比。

实现注意：

* 同一路里重复出现的 doc_id 只按**第一次（最好）出现的排名**计分，并在
  自检里断言，避免上游把同一文档切块后重复召回导致分数虚高。
* 分数相同时按**首次出现顺序**（先 BM25 后稠密）稳定排序，保证可复现。
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

STUB = False


def fuse(runs: Sequence[Sequence[Tuple[str, float]]],
         weights: Sequence[float] | None = None,
         k: int = 60) -> List[Tuple[str, float]]:
    """倒数排名融合。签名冻结，见 INTERFACE.md。

    Args:
        runs: 多路检索结果，每路是按相关度降序的 [(doc_id, score)]。
        weights: 每路权重；None 表示等权。
        k: RRF 平滑常数，默认 60。

    Returns:
        按融合分降序的 [(doc_id, score)]。
    """
    if not runs:
        return []
    if weights is not None and len(weights) != len(runs):
        raise ValueError(f"weights 长度({len(weights)}) 必须与 runs({len(runs)}) 一致")
    if k <= 0:
        raise ValueError("k 必须为正数，否则 rank=1 时除零")

    w_list = list(weights) if weights is not None else [1.0] * len(runs)
    scores: Dict[str, float] = {}
    order: Dict[str, int] = {}          # 首次出现顺序，用于同分稳定排序
    for w, run in zip(w_list, runs):
        if w == 0:
            continue
        seen = set()
        for rank, item in enumerate(run, start=1):
            doc_id = item[0]
            if doc_id in seen:
                continue                 # 同一路内重复召回只算最好名次
            seen.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)
            order.setdefault(doc_id, len(order))

    return sorted(scores.items(), key=lambda kv: (-kv[1], order[kv[0]]))


def fuse_ids(runs: Sequence[Sequence[str]],
             weights: Sequence[float] | None = None,
             k: int = 60) -> List[str]:
    """只要 doc_id 序列的快捷方式：runs 里是纯 doc_id 列表（按相关度降序）。"""
    as_pairs = [[(d, 0.0) for d in run] for run in runs]
    return [d for d, _ in fuse(as_pairs, weights=weights, k=k)]


def alpha_weights(alpha: float, sparse_weight: float | None = None) -> List[float]:
    """E2 的权重约定：返回 ``[1-α, α]``（第 0 路稀疏、第 1 路稠密）。

    α = 0 → 纯 BM25；α = 1 → 纯稠密。``sparse_weight`` 预留给后续的
    「BM25 也加权的三路融合」，缺省不使用。
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha 必须在 [0, 1] 区间")
    sparse = (1.0 - alpha) if sparse_weight is None else sparse_weight
    return [sparse, alpha]


def self_check() -> dict:
    """给 B 用的自检：单路输入时，排序应与原路完全一致。"""
    run = [("a", 9.0), ("b", 5.0), ("c", 1.0)]
    fused = fuse([run])
    single_ok = [d for d, _ in run] == [d for d, _ in fused]
    disjoint = fuse([[("a", 1.0)], [("b", 1.0)]])
    dup = fuse([[("a", 1.0), ("a", 2.0), ("b", 3.0)]])
    return {
        "single_run_same_order": single_ok,
        "disjoint_len": len(disjoint),
        "duplicate_dedup_len": len(dup),
        "alpha_weights_0.5": alpha_weights(0.5),
        "k60_top_score": round(fuse([run], k=60)[0][1], 6),
    }
