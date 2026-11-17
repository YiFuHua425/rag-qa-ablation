"""指标计算 —— 负责人：D（数据与评测）

状态：[READY] A 已给参考实现，**D 负责校验、补边界用例、加置信区间**。

全组只准用这一份。各自手写指标 = 报告里出现两套口径 = 数字对不上。

口径约定（写进报告「评测协议」小节）：
  * ranked 是按相关度**降序**的 doc_id 列表，长度可小于 k
  * 只有一个金标段落（CMRC2018 dev 的设定），nDCG 的理想值恒为 1
  * 中文 F1 按**字符级**切分（CMRC 官方口径），EM 用归一化后全串相等
  * 归一化：小写 → 去中英文标点 → 去所有空白
"""

from __future__ import annotations

import math
import random
import re
import unicodedata

# ---------------------------------------------------------------- 归一化

# 中英文标点 + 符号，统一删掉
_PUNCT = set("！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」"
             "『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—''‛""„‟…‧﹏")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """小写 + 去标点 + 去空白。中英文共用。"""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(ch for ch in text if ch not in _PUNCT)
    return _WS.sub("", text)


def _tokens(text: str) -> list[str]:
    """字符级切分（中文 QA 的官方口径）。"""
    return list(normalize(text))


# ---------------------------------------------------------------- 检索指标

def recall_at_k(ranked: list[str], gold: str, k: int) -> float:
    """前 k 名里有没有金标。单金标 → 取值 {0.0, 1.0}。"""
    if k <= 0 or not ranked:
        return 0.0
    return 1.0 if gold in ranked[:k] else 0.0


def mrr(ranked: list[str], gold: str) -> float:
    """金标排名倒数。第 1 名 = 1.0，第 5 名 = 0.2，未命中 = 0.0。"""
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id == gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: list[str], gold: str, k: int) -> float:
    """二值相关的 nDCG@k。单金标时 IDCG=1，故 nDCG = DCG。"""
    if k <= 0:
        return 0.0
    for rank, doc_id in enumerate(ranked[:k], start=1):
        if doc_id == gold:
            return 1.0 / math.log2(rank + 1)
    return 0.0


# ---------------------------------------------------------------- 答案指标

def em(pred: str, ref: str) -> float:
    """Exact Match（归一化后全串相等）。"""
    return 1.0 if normalize(pred) == normalize(ref) else 0.0


def f1(pred: str, ref: str) -> float:
    """字符级 F1（CMRC 官方口径）。"""
    p_toks, r_toks = _tokens(pred), _tokens(ref)
    if not p_toks and not r_toks:
        return 1.0
    if not p_toks or not r_toks:
        return 0.0
    common = 0
    ref_pool = list(r_toks)
    for tok in p_toks:
        if tok in ref_pool:
            common += 1
            ref_pool.remove(tok)
    if common == 0:
        return 0.0
    precision = common / len(p_toks)
    recall = common / len(r_toks)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------- 统计

def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def bootstrap_ci(values: list[float], n: int = 1000, alpha: float = 0.05,
                 seed: int = 42) -> tuple[float, float]:
    """均值的 bootstrap 95% 置信区间。返回 (lo, hi)。

    报告里所有主结果都要带这个区间，否则「提升 0.3 个点」会被问显著性。
    """
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    size = len(values)
    means = []
    for _ in range(n):
        sample = [values[rng.randrange(size)] for _ in range(size)]
        means.append(sum(sample) / size)
    means.sort()
    lo = means[int((alpha / 2) * n)]
    hi = means[min(n - 1, int((1 - alpha / 2) * n))]
    return (lo, hi)


# ---------------------------------------------------------------- 自检

def self_check() -> dict:
    """A 提供的自检用例；D 请在此基础上补边界（空列表、重复词、大小写等）。"""
    assert recall_at_k(["a", "b"], "b", 2) == 1.0
    assert recall_at_k(["a", "b"], "b", 1) == 0.0
    assert abs(mrr(["a", "b"], "b") - 0.5) < 1e-9
    assert abs(ndcg_at_k(["a", "b"], "b", 2) - 1 / math.log2(3)) < 1e-9
    assert em("北京。", "北京") == 1.0
    assert abs(f1("北京 大学", "北京大学") - 1.0) < 1e-9
    lo, hi = bootstrap_ci([1.0, 1.0, 0.0, 1.0], n=200)
    assert 0.0 <= lo <= hi <= 1.0
    return {"ok": True, "cases": 7}