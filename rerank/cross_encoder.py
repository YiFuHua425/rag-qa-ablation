"""交叉编码器重排 —— 负责人：C（重排与生成）

状态：[STUB] 空实现，等 C 替换为真实现。

要点：
  * 与双塔（bi-encoder）不同，交叉编码器把 (query, passage) 拼成一条输入喂给模型，
    能建模词级交互，精度高但**不能预计算**，所以只能对少量候选重排
  * `depth` 决定重排多少条：E4 的变量就是 depth ∈ {20, 50, 100}
  * 必须记录重排耗时，E4 要画「精度–延迟」Pareto 图
  * 延迟口径：**CPU 单线程、batch=1、重复 3 次取中位数**（config.yaml 的 latency 段）

实现清单（C 逐条打勾）：
  [ ] 懒加载模型（--dry-run 不许下载权重）
  [ ] 输入拼装：f"{query}</s>{passage}"（按模型 card 的模板）
  [ ] truncation=True, max_length=512
  [ ] 返回新的 [(doc_id, score)]，score 为 logit（仅排序用，不与检索分混算）
  [ ] 已按 score 降序，截断到 topn
  [ ] 记录 elapsed_ms 到结果 json
"""

from __future__ import annotations

import warnings

STUB = True
_WARNED = False


def _warn_once() -> None:
    global _WARNED
    if not _WARNED:
        warnings.warn(
            "rerank/cross_encoder.py 仍是 STUB，原样返回候选；C 请替换为真实现。",
            RuntimeWarning,
            stacklevel=2,
        )
        _WARNED = True


class Reranker:
    """交叉编码器重排器。签名冻结，见 INTERFACE.md。"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base",
                 depth: int = 50, topn: int = 5,
                 device: str = "cuda", max_length: int = 512) -> None:
        self.model_name = model_name
        self.depth = depth
        self.topn = topn
        self.device = device
        self.max_length = max_length
        self._model = None
        self.last_elapsed_ms: float = 0.0

    def _ensure_model(self):
        # TODO(C):
        # from transformers import AutoTokenizer, AutoModelForSequenceClassification
        _warn_once()
        raise NotImplementedError(
            "Reranker._ensure_model() 待 C 实现（真实运行需要 torch + transformers）"
        )

    def rerank(self, query: str,
               candidates: list[tuple[str, float]],
               topn: int | None = None) -> list[tuple[str, float]]:
        """对前 `depth` 个候选重排，返回前 `topn` 个。

        STUB 行为：**原样返回前 topn 条**（保持检索顺序），使下游能跑通。
        C 实现后，同样输入应给出不同排序（这就是 E4 的对照）。
        """
        _warn_once()
        n = topn if topn is not None else self.topn
        return list(candidates[:n])