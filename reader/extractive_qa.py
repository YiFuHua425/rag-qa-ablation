"""抽取式阅读器 —— 负责人：C（重排与生成）

状态：[STUB] 空实现，等 C 替换为真实现。

为什么主指标用**抽取式**而不是生成式：
  * 确定性、可复现，同一输入永远同一输出 → 实验对照干净
  * 8G 显存跑得动，3 天内不用调 prompt
  * 生成式（Qwen2.5-7B Q4）只做视频里的定性演示，**不承担主结论**

实现清单（C 逐条打勾）：
  [ ] 懒加载 tokenizer + model（--dry-run 不许下载权重）
  [ ] context 超 max_length 要截断（截左边保留右边？统一截**右**，并在报告里说明）
  [ ] 起止 logits → 取 argmax 的 (start, end)，约束 end ≥ start
  [ ] 无答案 / 分数低于 null_score → 返回 ("", 0.0)
  [ ] 返回原文片段（用 offset_mapping 还原到原字符串），不要返回 token 串
  [ ] 单条与 batch 两条路径都给，pipeline 里用 batch 快 5–10 倍
"""

from __future__ import annotations

import warnings

STUB = True
_WARNED = False


def _warn_once() -> None:
    global _WARNED
    if not _WARNED:
        warnings.warn(
            "reader/extractive_qa.py 仍是 STUB，返回空答案；C 请替换为真实现。",
            RuntimeWarning,
            stacklevel=2,
        )
        _WARNED = True


class ExtractiveReader:
    """抽取式问答。签名冻结，见 INTERFACE.md。"""

    def __init__(self, model_name: str = "uer/roberta-base-chinese-extractive-qa",
                 max_length: int = 512, device: str = "cuda",
                 batch_size: int = 16, null_score: float = -100.0) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.device = device
        self.batch_size = batch_size
        self.null_score = null_score
        self._model = None
        self._tok = None

    def _ensure_model(self):
        # TODO(C):
        # from transformers import AutoTokenizer, AutoModelForQuestionAnswering
        _warn_once()
        raise NotImplementedError(
            "ExtractiveReader._ensure_model() 待 C 实现（真实运行需要 torch + transformers）"
        )

    def answer(self, question: str, context: str) -> tuple[str, float]:
        """返回 (answer_text, score)；无答案返回 ("", 0.0)。

        STUB 行为：返回空答案，使 pipeline 能跑通而不崩。
        """
        _warn_once()
        return "", 0.0

    def answer_batch(self, pairs: list[tuple[str, str]]) -> list[tuple[str, float]]:
        """批量版：pairs = [(question, context), ...]。STUB 同 answer()。"""
        _warn_once()
        return [("", 0.0) for _ in pairs]