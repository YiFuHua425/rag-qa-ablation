"""上下文拼装 —— 负责人：C（重排与生成）

状态：[READY]（第 2/5 棒：朴素版 + E5 位置控制）

**E5 位置偏差实验只改这一个函数。** 把金标段落放到上下文的开头 / 中间 / 末尾，
看答案抽取的 F1 怎么掉——这是全案唯一「不动模型也能出洞察」的实验。

规格（INTERFACE.md 第 2 节冻结）：
  * ``order = "natural"``      保持检索顺序（默认，正常实验用这个）
  * ``order = "gold_first"``   把金标段落挪到第一位
  * ``order = "gold_middle"``  把金标段落挪到正中间
  * ``order = "gold_last"``    把金标段落挪到最后一位
  * ``max_chars`` 截断策略：**按段落整段拼入，超限即停**（不许截半个段落，
    否则 F1 的差异会混入"句子被切断"这个混淆因素）
  * 返回「实际放进上下文的 doc_id 列表」，供 E5 统计"金标是否被保留"

三个实现细节值得写进报告：

1. **doc_map 兼容两种形状**：``{doc_id: text}`` 与 ``{doc_id: {"doc_id","text"}}``。
   仓库里两处调用方给的形状不一致（``run_e1.run_real`` 传的是后者），
   这里统一在入口归一化，避免 ``"\n".join()`` 拿到 dict 直接 TypeError。
2. **诊断信息不塞进返回值**：返回值必须是二元组（冻结接口，别处按两元解包），
   所以每次调用后把本轮统计放进模块级 ``LAST_STATS``，用 :func:`get_last_stats` 取。
   E5 靠它区分「金标根本没拼进上下文（截断/排序）」与「金标进去了但阅读器没答对」。
3. **至少保留第一段**：即使第一段单独就超过 ``max_chars`` 也拼进去，否则上下文为空、
   阅读器必然返回空答案，会把"超长段落"混淆成"模型不会答"。溢出量记在
   ``overflow_chars`` 里，报告里如实说明。
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

STUB = False

_VALID_ORDER = ("natural", "gold_first", "gold_middle", "gold_last")

#: 最近一次 :func:`build_context` 的诊断信息（单线程使用；E5 每次调用后读一次）。
LAST_STATS: Dict[str, object] = {}


def resolve_text(value) -> str | None:
    """把 doc_map 的值归一成纯文本；认不出返回 ``None``。"""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        text = value.get("text")
        return text if isinstance(text, str) else None
    return None


def order_doc_ids(doc_ids: Sequence[str], order: str = "natural",
                  gold_doc_id: str | None = None) -> List[str]:
    """按 ``order`` 重排 doc_id（不改原列表）。

    金标不在候选里、或 ``order="natural"`` 时原样返回——E5 的
    ``gold_in_candidates=0`` 那批题会被单独统计，不参与位置效应。
    """
    ids = list(doc_ids)
    if order == "natural" or not gold_doc_id or gold_doc_id not in ids:
        return ids
    ids.remove(gold_doc_id)
    if order == "gold_first":
        pos = 0
    elif order == "gold_last":
        pos = len(ids)
    else:                                   # gold_middle：正中间（5 段 → 第 3 位）
        pos = len(ids) // 2
    ids.insert(pos, gold_doc_id)
    return ids


def build_context(doc_ids: List[str],
                  doc_map: dict,
                  order: str = "natural",
                  max_chars: int = 2000,
                  gold_doc_id: str | None = None) -> tuple[str, list[str]]:
    """拼装上下文。签名冻结，见 INTERFACE.md。

    Args:
        doc_ids: 检索返回的 doc_id，按相关度降序。
        doc_map: ``{doc_id: text}``（也接受 ``{doc_id: {"text": ...}}``）
        order: natural | gold_first | gold_middle | gold_last
        max_chars: 上下文长度上限（字符）
        gold_doc_id: 金标段落（仅 E5 的 order != natural 时需要）

    Returns:
        ``(context_text, used_doc_ids)`` —— used_doc_ids 是真正进入上下文的段落，
        E5 用它判断「金标是否被保留」。

    Raises:
        ValueError: ``order`` 不在四档之内。
    """
    if order not in _VALID_ORDER:
        raise ValueError(f"order 必须是 {_VALID_ORDER} 之一，收到 {order!r}")
    if max_chars <= 0:
        raise ValueError(f"max_chars 必须为正，收到 {max_chars!r}")

    candidates = list(dict.fromkeys(doc_ids))          # 去重但保序：重复 id 会白占额度
    ordered = order_doc_ids(candidates, order, gold_doc_id)
    gold_in_candidates = bool(gold_doc_id) and gold_doc_id in candidates

    parts: List[str] = []
    used: List[str] = []
    missing: List[str] = []
    total = 0
    stop_at: int | None = None                          # 第一个装不下的位置

    for pos, doc_id in enumerate(ordered):
        text = resolve_text(doc_map.get(doc_id))
        if not text:
            missing.append(doc_id)
            continue
        # 整段拼入：装不下就停，绝不截半个段落（否则 F1 混入"句子被切断"）
        if parts and total + len(text) + 1 > max_chars:
            stop_at = pos
            break
        parts.append(text)
        used.append(doc_id)
        total += len(text) + 1

    dropped = ordered[stop_at:] if stop_at is not None else []
    gold_pos = (used.index(gold_doc_id) + 1) if gold_in_candidates and gold_doc_id in used else None

    LAST_STATS.clear()
    LAST_STATS.update({
        "order": order,
        "gold_doc_id": gold_doc_id,
        "gold_in_candidates": gold_in_candidates,
        "gold_used": gold_pos is not None,
        "gold_position": gold_pos,                     # 1 起；None = 没进上下文
        "n_candidates": len(candidates),
        "n_used": len(used),
        "chars": total,
        "max_chars": max_chars,
        "overflow_chars": max(0, total - max_chars),
        "dropped_doc_ids": dropped,                    # 被 max_chars 挡在门外的
        "missing_doc_ids": missing,                    # doc_map 里查不到的
    })
    return "\n".join(parts), used


def get_last_stats() -> Dict[str, object]:
    """最近一次 :func:`build_context` 的诊断信息（副本，随便改）。"""
    return dict(LAST_STATS)


# ------------------------------------------------------------------ 自检

def self_check() -> dict:
    """不依赖模型与数据的小自检，``python -m reader.context_builder`` 直接跑。"""
    docs = {"a": "甲" * 10, "b": "乙" * 10, "c": "丙" * 10, "d": "丁" * 10, "e": "戊" * 10}
    ids = ["a", "b", "c", "d", "e"]

    ctx, used = build_context(ids, docs, "natural", 1000)
    assert used == ids and ctx == "\n".join(docs[i] for i in ids)

    ctx, used = build_context(ids, docs, "gold_first", 1000, gold_doc_id="d")
    assert used[0] == "d", used

    ctx, used = build_context(ids, docs, "gold_last", 1000, gold_doc_id="b")
    assert used[-1] == "b", used

    ctx, used = build_context(ids, docs, "gold_middle", 1000, gold_doc_id="a")
    assert used[len(used) // 2] == "a", used
    assert get_last_stats()["gold_position"] == len(used) // 2 + 1

    # 截断：整段拼入，超限即停（5 段 × 11 字 = 55，上限 30 → 只放 2 段）
    ctx, used = build_context(ids, docs, "natural", 30)
    assert used == ["a", "b"], used
    assert get_last_stats()["dropped_doc_ids"] == ["c", "d", "e"]

    # 金标被截断挡掉 → gold_used=False（E5 靠这个区分"没读到"与"读错"）
    _, used = build_context(ids, docs, "gold_last", 30, gold_doc_id="e")
    st = get_last_stats()
    assert st["gold_in_candidates"] and not st["gold_used"] and used == ["a", "b"]

    # doc_map 给成 {"text": ...} 也能用；缺失 id 跳过不报错
    docs2 = {k: {"doc_id": k, "text": v} for k, v in docs.items()}
    ctx, used = build_context(["zzz", "a"], docs2, "natural", 1000)
    assert used == ["a"] and get_last_stats()["missing_doc_ids"] == ["zzz"]

    # 单段超长：至少保留第一段，溢出量记账
    _, used = build_context(["long"], {"long": "x" * 500}, "natural", 100)
    st = get_last_stats()
    assert used == ["long"] and st["overflow_chars"] == 501 - 100

    try:
        build_context(ids, docs, "gold_side", 100)
    except ValueError:
        pass
    else:
        raise AssertionError("非法 order 应该抛 ValueError")

    return {"ok": True, "cases": 10}


if __name__ == "__main__":
    print(self_check())
