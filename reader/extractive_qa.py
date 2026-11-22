"""抽取式阅读器 —— 负责人：C（重排与生成）

状态：[READY]（第 2 棒：单条跑通；第 3 棒：batch + 长上下文滑窗）

为什么主指标用**抽取式**而不是生成式：
  * 确定性、可复现，同一输入永远同一输出 → 实验对照干净
  * 显存/内存小，不用调 prompt
  * 生成式（Qwen2.5-7B Q4）只做视频里的定性演示，**不承担主结论**

实现要点（报告第 4 章「实现过程」的素材）：

1. **懒加载**：``_ensure_model()`` 才 import torch / transformers 并读权重，
   ``--dry-run`` 与 ``--list`` 不碰模型。
2. **长上下文滑窗**：``truncation="only_second"`` + ``max_length`` + ``doc_stride``，
   一篇 2000 字的上下文切成多个 feature 分别打分，再取全局最高分的 span。
   CMRC 段落均值 509 字（≈700 token），**不滑窗会有一半答案落在窗外**。
3. **只用 context 侧的 token**：``sequence_ids == 1`` 过滤掉问题与特殊符号，
   否则模型可能返回问题里的词（早期版本真踩过，F1 虚高）。
4. **offset 还原原文**：用 ``offset_mapping`` 把 token 下标映射回字符区间，
   返回原文片段而不是 token 串——中文 BERT 分词是字级的，直接拼 token 会多空格。
5. **score 口径**：``start_logit + end_logit``（未归一化），只用于同一题内的候选比较；
   跨题不可比，也不与检索分混算。
6. **无答案**：``score < null_score`` 或还原出空串 → ``("", 0.0)``。
   本项目的模型是 SQuAD1 风格（必答），所以 ``null_score`` 默认 -100（等于不用），
   留成配置项是为了换 SQuAD2 风格权重时不用改代码。
"""

from __future__ import annotations

import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

STUB = False

_IMPORT_HINT = (
    "阅读器需要 torch + transformers：\n"
    "  pip install --index-url https://download.pytorch.org/whl/cpu "
    "--extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple torch transformers\n"
    "权重：models/roberta-base-chinese-extractive-qa（见 scripts/download_models_c.py），"
    "或用 --reader-model 指定本地目录"
)


class ExtractiveReader:
    """抽取式问答。签名冻结（INTERFACE.md 第 2 节）。"""

    def __init__(self, model_name: str = "uer/roberta-base-chinese-extractive-qa",
                 max_length: int = 512, device: str = "cuda",
                 batch_size: int = 16, null_score: float = -100.0,
                 model_path: str | None = None, doc_stride: int = 128,
                 max_answer_len: int = 50, n_best: int = 20,
                 threads: int = 1, score_threads: int = 0) -> None:
        self.model_name = model_name
        self.max_length = int(max_length)
        self.device = device
        self.batch_size = int(batch_size)
        self.null_score = float(null_score)
        self.model_path = model_path
        self.doc_stride = int(doc_stride)
        self.max_answer_len = int(max_answer_len)
        self.n_best = int(n_best)
        self.threads = int(threads)
        self.score_threads = int(score_threads)
        self._model = None
        self._tok = None
        self._torch = None
        self.last_stats: Dict[str, object] = {}

    # ------------------------------------------------------------ 懒加载
    def _ensure_model(self):
        if self._model is not None:
            return self._model
        try:
            import torch
            from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        except Exception as exc:  # pragma: no cover - 依赖缺失时才走到
            raise RuntimeError(_IMPORT_HINT) from exc

        path = self.model_path or self.model_name
        self._tok = AutoTokenizer.from_pretrained(path, use_fast=True)
        self._model = AutoModelForQuestionAnswering.from_pretrained(path)
        self._model.eval()
        if self.device != "cpu" and not torch.cuda.is_available():
            print(f"[reader] 没有可用的 {self.device}，自动退回 cpu", flush=True)
            self.device = "cpu"
        if self.threads > 0:
            torch.set_num_threads(self.threads)
        self._model.to(self.device)
        self._torch = torch
        return self._model

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _apply_threads(self, n: int) -> None:
        """临时切换 torch 线程数：算指标用多线程，报延迟用单线程（口径隔离）。"""
        if self._torch is not None and n and n > 0:
            self._torch.set_num_threads(int(n))

    # ------------------------------------------------------------ 单条
    def answer(self, question: str, context: str) -> tuple[str, float]:
        """返回 (answer_text, score)；无答案返回 ("", 0.0)。"""
        return self.answer_batch([(question, context)])[0]

    # ------------------------------------------------------------ 批量
    def _features(self, pairs: Sequence[Tuple[str, str]]):
        """一个问题/上下文 → 若干 feature（滑窗切片），并记录归属。"""
        questions = [q for q, _ in pairs]
        contexts = [c for _, c in pairs]
        enc = self._tok(
            questions,
            contexts,
            truncation="only_second",
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_tensors=None,
        )
        n_features = len(enc["input_ids"])
        mapping = enc.get("overflow_to_sample_mapping") or [i for i in range(len(contexts))]
        return enc, list(mapping), n_features

    def _best_span(self, start_logits: np.ndarray, end_logits: np.ndarray,
                   offsets, seq_ids) -> Tuple[float, int, int]:
        """单个 feature 内挑 (score, start_char, end_char)。"""
        n = len(start_logits)
        order = np.argsort(-start_logits)[: self.n_best]
        best = (float("-inf"), 0, 0)
        for s in order:
            s = int(s)
            if seq_ids[s] != 1:                       # 问题侧 / 特殊符号，跳过
                continue
            e_limit = min(n, s + self.max_answer_len)
            e = int(np.argmax(end_logits[s:e_limit])) + s
            if seq_ids[e] != 1:
                continue
            score = float(start_logits[s] + end_logits[e])
            if score > best[0]:
                best = (score, s, e)
        return best

    def answer_batch(self, pairs: list[tuple[str, str]]) -> list[tuple[str, float]]:
        """批量版：pairs = [(question, context), ...]。"""
        if not pairs:
            return []
        self._ensure_model()
        torch = self._torch
        self._apply_threads(self.score_threads)
        t0 = time.perf_counter()
        enc, mapping, n_features = self._features(pairs)

        # feature 级批处理：跨题一起前向，比逐题快 5–10 倍
        input_ids = enc["input_ids"]
        attention = enc["attention_mask"]
        token_type = enc.get("token_type_ids")
        results: List[Tuple[str, float]] = [("", 0.0) for _ in pairs]
        best_scores: List[float] = [float("-inf")] * len(pairs)
        answers: List[str] = [""] * len(pairs)
        contexts = [c for _, c in pairs]

        for lo in range(0, n_features, self.batch_size):
            hi = min(n_features, lo + self.batch_size)
            batch = {
                "input_ids": torch.tensor(input_ids[lo:hi], dtype=torch.long, device=self.device),
                "attention_mask": torch.tensor(attention[lo:hi], dtype=torch.long, device=self.device),
            }
            if token_type is not None:
                batch["token_type_ids"] = torch.tensor(token_type[lo:hi], dtype=torch.long,
                                                       device=self.device)
            with torch.no_grad():
                out = self._model(**batch)
            starts = out.start_logits.detach().cpu().numpy()
            ends = out.end_logits.detach().cpu().numpy()

            for j in range(hi - lo):
                fidx = lo + j
                sample = mapping[fidx]
                offsets = enc["offset_mapping"][fidx]
                seq_ids = enc.sequence_ids(fidx)
                score, s, e = self._best_span(starts[j], ends[j], offsets, seq_ids)
                if not np.isfinite(score) or score < self.null_score:
                    continue
                cs, ce = offsets[s][0], offsets[e][1]
                text = contexts[sample][cs:ce].strip()
                if not text:
                    continue
                if score > best_scores[sample]:
                    best_scores[sample] = score
                    answers[sample] = text

        for i, (text, _score) in enumerate(zip(answers, best_scores)):
            results[i] = (text, float(_score)) if text else ("", 0.0)

        self.last_stats = {
            "n_pairs": len(pairs),
            "n_features": n_features,
            "features_per_pair": round(n_features / max(1, len(pairs)), 2),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 3),
            "device": self.device,
            "model": self.model_path or self.model_name,
        }
        return results

    # ------------------------------------------------------------ 延迟
    def benchmark_latency(self, pairs: Sequence[Tuple[str, str]], repeat: int = 3) -> float:
        """单条延迟（batch=1、CPU 单线程口径，重复 ``repeat`` 次取中位数），单位 ms。"""
        if not pairs:
            return 0.0
        self._ensure_model()
        self.batch_size, keep = 1, self.batch_size
        self._apply_threads(self.threads)                 # 正式口径：单线程
        times = []
        try:
            for _ in range(repeat):
                for q, c in pairs:
                    t0 = time.perf_counter()
                    self.answer(q, c)
                    times.append((time.perf_counter() - t0) * 1000.0)
        finally:
            self.batch_size = keep
            self._apply_threads(self.score_threads or self.threads)
        times.sort()
        return round(times[len(times) // 2], 3)

    def __repr__(self) -> str:
        return (f"ExtractiveReader({self.model_path or self.model_name}, "
                f"device={self.device}, max_len={self.max_length}, stride={self.doc_stride})")
