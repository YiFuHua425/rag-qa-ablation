"""检索实验的评测封装 —— 负责人：B（只调用 ``eval/metrics.py``，不自己实现指标）

INTERFACE.md 规定「全组只准用这一份指标」，所以这里只做**批量聚合**：
把一行行 ``ranked doc_id 列表`` 摊平成 ``recall@5`` / ``mrr`` / ``ndcg@5``
这一套字典，并附 bootstrap 95% 置信区间。

口径提醒（写进报告「评测协议」）：

* 单金标段落，``recall@k`` 取值 {0,1}，均值即命中率；
* ``mrr`` 不设 k 上限（金标排到第 50 名也算分，只是很小）；
* ``ndcg@k`` 单金标时 IDCG=1，故 nDCG = DCG；
* 延迟只统计**检索 + 融合**（不含编码与建索引），口径与 config ``latency`` 一致。
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from . import metrics as M


def metrics_dict(rankings: Sequence[Sequence[str]], golds: Sequence[str], *,
                 ks: Sequence[int] = (1, 5, 10, 20), ndcg_k: int = 5) -> Dict[str, float]:
    """批量算 Recall@k / MRR / nDCG@k / n。"""
    n = len(rankings)
    if n != len(golds):
        raise ValueError(f"rankings({n}) 与 golds({len(golds)}) 长度不一致")
    out: Dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = (
            sum(M.recall_at_k(list(r), g, k) for r, g in zip(rankings, golds)) / n if n else 0.0
        )
    out["mrr"] = sum(M.mrr(list(r), g) for r, g in zip(rankings, golds)) / n if n else 0.0
    out[f"ndcg@{ndcg_k}"] = (
        sum(M.ndcg_at_k(list(r), g, ndcg_k) for r, g in zip(rankings, golds)) / n if n else 0.0
    )
    out["n_questions"] = n
    # INTERFACE 第 3 节的结果格式里 em/f1 是必现字段：检索类 run 不跑阅读器，
    # 这里显式记 None（而不是缺键），下游读表时才不会 KeyError 或误读成 0。
    out["em"] = None
    out["f1"] = None
    return out


def per_query_recall(rankings: Sequence[Sequence[str]], golds: Sequence[str], k: int = 5) -> List[float]:
    """每题 Recall@k（0/1），喂给 ``bootstrap_ci``。"""
    return [M.recall_at_k(list(r), g, k) for r, g in zip(rankings, golds)]


def per_query_mrr(rankings: Sequence[Sequence[str]], golds: Sequence[str]) -> List[float]:
    return [M.mrr(list(r), g) for r, g in zip(rankings, golds)]


def recall_ci(rankings: Sequence[Sequence[str]], golds: Sequence[str], k: int = 5,
              n_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> List[float]:
    lo, hi = M.bootstrap_ci(per_query_recall(rankings, golds, k), n=n_boot, alpha=alpha, seed=seed)
    return [round(lo, 6), round(hi, 6)]


def per_question_records(qids: Sequence[str], rankings: Sequence[Sequence[str]],
                         golds: Sequence[str], top_show: int = 10) -> List[dict]:
    """INTERFACE.md 第 3 节的 ``per_question``：只存前 ``top_show`` 个排名，控制文件体积。"""
    rows: List[dict] = []
    for qid, ranked, gold in zip(qids, rankings, golds):
        ranked = list(ranked)
        hit = gold in ranked
        rows.append({
            "qid": qid,
            "ranked": ranked[:top_show],
            "gold": gold,
            "hit": bool(hit),
            "rank": (ranked.index(gold) + 1) if hit else None,
            # 检索类 run 不跑阅读器：按 INTERFACE 第 3 节保留字段，值为空
            "pred": "",
            "f1": None,
        })
    return rows


def format_line(tag: str, met: Dict[str, float], ci: Sequence[float] | None = None) -> str:
    ci_txt = f" [{ci[0]:.4f}, {ci[1]:.4f}]" if ci else ""
    return (f"{tag:22s} R@1={met.get('recall@1', 0):.4f} R@5={met.get('recall@5', 0):.4f}{ci_txt} "
            f"MRR={met.get('mrr', 0):.4f} nDCG@5={met.get('ndcg@5', 0):.4f}")
