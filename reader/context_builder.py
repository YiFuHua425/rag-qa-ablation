"""上下文拼装 —— 负责人：C（重排与生成）

状态：[STUB] 空实现，等 C 替换为真实现。

**E5 位置偏差实验只改这一个函数。**
把金标段落放到上下文的开头 / 中间 / 末尾，看答案抽取的 F1 怎么掉——
这是全案唯一「不动模型也能出洞察」的实验。

规格：
  * order = "natural"      保持检索顺序（默认，正常实验用这个）
  * order = "gold_first"   把金标段落挪到第一位
  * order = "gold_middle"  把金标段落挪到正中间
  * order = "gold_last"    把金标段落挪到最后一位
  * max_chars 截断策略：**按段落整段拼入，超限即停**（不许截半个段落，
    否则 F1 的差异会混入"句子被切断"这个混淆因素）
  * 必须返回「实际放进上下文的 doc_id 列表」，供 E5 统计"金标是否被保留"

实现清单（C 逐条打勾）：
  [ ] doc_map 缺失的 doc_id 跳过并计数（不要 KeyError）
  [ ] 截断后如果金标段落没进上下文，要能在返回值里体现（E5 需要区分
      "没读到" 与 "读到了但没答对"）
  [ ] 段落之间用 "\\n" 分隔，不加额外标题（避免引入格式混淆因素）
"""

from __future__ import annotations

import warnings

STUB = True
_WARNED = False

_VALID_ORDER = ("natural", "gold_first", "gold_middle", "gold_last")


def _warn_once() -> None:
    global _WARNED
    if not _WARNED:
        warnings.warn(
            "reader/context_builder.py 仍是 STUB；C 请替换为真实现。",
            RuntimeWarning,
            stacklevel=2,
        )
        _WARNED = True


def build_context(doc_ids: list[str],
                  doc_map: dict,
                  order: str = "natural",
                  max_chars: int = 2000,
                  gold_doc_id: str | None = None) -> tuple[str, list[str]]:
    """拼装上下文。签名冻结，见 INTERFACE.md。

    Args:
        doc_ids: 检索返回的 doc_id，按相关度降序。
        doc_map: {doc_id: text}
        order: natural | gold_first | gold_middle | gold_last
        max_chars: 上下文长度上限（字符）
        gold_doc_id: 金标段落（仅 E5 的 order != natural 时需要）

    Returns:
        (context_text, used_doc_ids) —— used_doc_ids 是真正进入上下文的段落，
        E5 用它判断「金标是否被保留」。

    STUB 行为：按给定顺序整段拼接、按 max_chars 截断，能跑通但不做位置重排。
    """
    _warn_once()
    if order not in _VALID_ORDER:
        raise ValueError(f"order 必须是 {_VALID_ORDER} 之一，收到 {order!r}")

    # TODO(C): 按 order 重排 doc_ids（gold_first/middle/last），再拼装
    parts: list[str] = []
    used: list[str] = []
    total = 0
    for doc_id in doc_ids:
        text = doc_map.get(doc_id)
        if not text:
            continue
        if total + len(text) > max_chars and parts:
            break
        parts.append(text)
        used.append(doc_id)
        total += len(text) + 1
    return "\n".join(parts), used