"""交叉编码器重排 —— 负责人：C（重排与生成）

状态：[READY]（第 3 棒：真实现 + 开关 + 延迟口径；第 4 棒：E4 的 depth 扫描）

要点（报告第 4 章「实现过程」的素材）：
  * 与双塔（bi-encoder）不同，交叉编码器把 (query, passage) 拼成**一条**输入喂给模型，
    能建模词级交互，精度高但**不能预计算**，所以只能对少量候选重排
  * `depth` 决定重排多少条：E4 的变量就是 depth ∈ {20, 50, 100}
  * 必须记录重排耗时，E4 要画「精度–延迟」Pareto 图
  * 延迟口径：**CPU 单线程、batch=1、重复 3 次取中位数**（config.yaml 的 latency 段）
    ——吞吐实验用 ``batch_size`` 批量跑，但报出来的延迟一律走 :meth:`benchmark_latency`

三个实现细节：

1. **签名只给 (doc_id, score)，文本得自己带**：INTERFACE 冻结的 ``rerank`` 拿不到段落原文，
   所以本类用 ``doc_map``（``set_doc_map`` / ``set_corpus`` / 构造参数）解析文本。
   解析不到的候选**不丢**，按原顺序排在末尾、score 记 ``-inf``——
   E4 的 Recall@5 是在重排后的前 5 名上算的，丢掉候选等于偷偷改了召回率。
2. **logit 只用于排序**：返回的 score 是模型 logit（``num_labels=2`` 时取正类减负类），
   与检索分不同量纲，**不许和 RRF 分数相加**（加了就是两套分数混算，报告里会被问）。
3. **三种后端**：``hf``（transformers，正式口径）、``onnx``（onnxruntime，没有 torch 的机器）、
   ``lexical``（**离线占位，数字不可进报告**，只为让没装模型的机器能跑通链路）。
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

STUB = False

PKG_ROOT = Path(__file__).resolve().parents[1]

_IMPORT_HINT = (
    "重排需要 torch + transformers（或先导出 ONNX）：\n"
    "  pip install --index-url https://download.pytorch.org/whl/cpu "
    "--extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple torch transformers\n"
    "权重：python scripts/download_models_c.py（下到 models/bge-reranker-base），"
    "或用 --rerank-model 指定本地目录"
)


def _resolve_text(value) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        text = value.get("text")
        return text if isinstance(text, str) else None
    return None


def _default_model_dir() -> Path:
    return PKG_ROOT / "models" / "bge-reranker-base"


class Reranker:
    """交叉编码器重排器。``rerank`` 签名冻结，见 INTERFACE.md。"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base",
                 depth: int = 50, topn: int = 5,
                 device: str = "cuda", max_length: int = 512,
                 model_path: str | None = None, batch_size: int = 16,
                 backend: str = "auto", threads: int = 1,
                 doc_map: dict | None = None, score_threads: int = 0) -> None:
        self.model_name = model_name
        self.depth = int(depth)
        self.topn = int(topn)
        self.device = device
        self.max_length = int(max_length)
        self.model_path = model_path
        self.batch_size = int(batch_size)
        self.backend = backend
        self.threads = int(threads)
        self.doc_map = doc_map
        self.score_threads = int(score_threads)
        self._model = None
        self._tok = None
        self._torch = None
        self._engine: str | None = None
        self.last_elapsed_ms: float = 0.0
        self.last_stats: Dict[str, object] = {}

    # ------------------------------------------------------------ 语料
    def set_doc_map(self, doc_map: dict) -> "Reranker":
        """``{doc_id: text}`` 或 ``{doc_id: {"text": ...}}``。"""
        self.doc_map = doc_map
        return self

    def set_corpus(self, docs: Sequence[dict]) -> "Reranker":
        """直接喂 ``corpus.jsonl`` 的行。"""
        self.doc_map = {d["doc_id"]: d["text"] for d in docs}
        return self

    # ------------------------------------------------------------ 懒加载
    def _model_dir(self) -> Path:
        return Path(self.model_path) if self.model_path else _default_model_dir()

    def _pick_backend(self) -> str:
        if self.backend != "auto":
            return self.backend
        try:                                     # 有 torch 就优先 transformers（正式口径）
            import torch  # noqa: F401
            import transformers  # noqa: F401

            return "hf"
        except Exception:
            pass
        path = self._model_dir()
        if path.is_dir() and any(path.glob("*.onnx")):
            return "onnx"
        raise RuntimeError(_IMPORT_HINT)

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        engine = self._pick_backend()
        path = str(self._model_dir() if self.model_path is None else self.model_path)

        if engine == "lexical":
            warnings.warn(
                "Reranker 使用 lexical 后端：这是**离线占位**，分数与真实交叉编码器无关，"
                "只用于验证链路；任何写进报告的 E4 数字都必须来自 hf/onnx 后端。",
                RuntimeWarning, stacklevel=2,
            )
            self._engine = "lexical"
            self._model = "lexical"
            return self._model

        if engine == "onnx":  # pragma: no cover - 只有没装 torch 的机器会走到
            import onnxruntime as ort
            from transformers import AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(path)
            so = ort.SessionOptions()
            threads = int(self.score_threads or self.threads or 0)
            if threads > 0:
                so.intra_op_num_threads = threads
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            onnx_file = next(iter(sorted(Path(path).glob("*.onnx"))))
            self._model = ort.InferenceSession(str(onnx_file), so, providers=["CPUExecutionProvider"])
            self._engine = "onnx"
            return self._model

        if engine == "hf":
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(path)
            self._model = AutoModelForSequenceClassification.from_pretrained(path)
            self._model.eval()
            if self.device != "cpu" and not torch.cuda.is_available():
                print(f"[rerank] 没有可用的 {self.device}，自动退回 cpu", flush=True)
                self.device = "cpu"
            if self.threads > 0:
                torch.set_num_threads(self.threads)
            self._model.to(self.device)
            self._torch = torch
            self._engine = "hf"
            return self._model

        raise ValueError(f"未知重排后端 {engine!r}")

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def engine(self) -> str:
        return self._engine or self.backend

    # ------------------------------------------------------------ 打分
    def _score_pairs(self, query: str, passages: Sequence[str],
                     batch_size: int | None = None) -> List[float]:
        """(query, passage) → logit。批量前向，顺序与输入一致。"""
        self._ensure_model()
        if not passages:
            return []
        bs = int(batch_size or self.batch_size)
        self._apply_threads(self.score_threads)

        if self._engine == "lexical":                       # 离线占位
            return [self._lexical_score(query, p) for p in passages]

        scores: List[float] = []
        if self._engine == "onnx":                          # pragma: no cover
            for lo in range(0, len(passages), bs):
                chunk = passages[lo:lo + bs]
                enc = self._tok([query] * len(chunk), list(chunk), padding=True,
                                truncation=True, max_length=self.max_length,
                                return_tensors="np")
                feed = {k: v.astype("int64") for k, v in enc.items()
                        if k in {i.name for i in self._model.get_inputs()}}
                logits = self._model.run(None, feed)[0]
                scores.extend(self._collapse_logits(logits))
            return scores

        torch = self._torch
        with torch.inference_mode():
            for lo in range(0, len(passages), bs):
                chunk = passages[lo:lo + bs]
                enc = self._tok([query] * len(chunk), list(chunk), padding=True,
                                truncation=True, max_length=self.max_length,
                                return_tensors="pt")
                enc = {k: v.to(self.device) for k, v in enc.items()}
                logits = self._model(**enc).logits.float().detach().cpu().numpy()
                scores.extend(self._collapse_logits(logits))
        return scores

    def _apply_threads(self, n: int) -> None:
        """临时切换 torch 线程数：算指标用多线程，报延迟用单线程（口径隔离）。"""
        if self._torch is not None and n and n > 0:
            self._torch.set_num_threads(int(n))

    @staticmethod
    def _collapse_logits(logits) -> List[float]:
        """(B,1) → 直接取； (B,2) → 正类减负类。"""
        import numpy as np

        arr = np.asarray(logits)
        if arr.ndim == 1:
            return [float(x) for x in arr]
        if arr.shape[1] == 1:
            return [float(x) for x in arr[:, 0]]
        return [float(x[1] - x[0]) for x in arr]

    @staticmethod
    def _lexical_score(query: str, passage: str) -> float:
        """字符二元组重合度。只给占位后端用，**不是**模型分数。"""
        def grams(text: str) -> set:
            text = "".join(ch for ch in text if ch.strip())
            return {text[i:i + 2] for i in range(len(text) - 1)} or set(text)

        q, p = grams(query), grams(passage)
        if not q or not p:
            return 0.0
        return len(q & p) / (len(q) ** 0.5 * len(p) ** 0.5)

    # ------------------------------------------------------------ 重排
    def score_candidates(self, query: str,
                         candidates: Sequence[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """对**全部**候选打分并按分数降序返回（E4 需要保留 depth 之外的原顺序）。"""
        t0 = time.perf_counter()
        pairs = list(candidates)
        texts, known = [], []
        for doc_id, _score in pairs:
            text = _resolve_text((self.doc_map or {}).get(doc_id))
            if text:
                texts.append(text)
                known.append(doc_id)

        scored = self._score_pairs(query, texts) if texts else []
        by_id = dict(zip(known, scored))
        out: List[Tuple[str, float]] = []
        missing: List[Tuple[str, float]] = []
        for doc_id, old in pairs:
            if doc_id in by_id:
                out.append((doc_id, by_id[doc_id]))
            else:
                missing.append((doc_id, float("-inf")))
        out.sort(key=lambda x: -x[1])                 # 稳定排序：同分保持检索顺序
        self.last_elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        self.last_stats = {
            "backend": self.engine,
            "depth": self.depth,
            "n_candidates": len(pairs),
            "n_scored": len(texts),
            "missing_text": [d for d, _ in missing],
            "elapsed_ms": self.last_elapsed_ms,
            "device": self.device,
        }
        return out + missing

    def rerank(self, query: str,
               candidates: list[tuple[str, float]],
               topn: int | None = None) -> list[tuple[str, float]]:
        """对前 ``depth`` 个候选重排，返回前 ``topn`` 个。

        超过 ``depth`` 的候选不参与重排（交叉编码器只在少量候选上才付得起代价），
        但仍然保留在返回列表的尾部，保证"重排不改变召回集合、只改顺序"。
        """
        n = int(topn if topn is not None else self.topn)
        head = list(candidates[: self.depth])
        tail = list(candidates[self.depth:])
        reranked = self.score_candidates(query, head)
        return (reranked + tail)[:n]

    # ------------------------------------------------------------ 延迟
    def benchmark_latency(self, query: str, candidates: Sequence[Tuple[str, float]],
                          repeat: int = 3) -> float:
        """单条重排延迟：batch=1、重复 ``repeat`` 次取中位数，单位 ms。"""
        if not candidates:
            return 0.0
        self._ensure_model()
        times = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            head = list(candidates[: self.depth])
            texts = [_resolve_text((self.doc_map or {}).get(d, "")) or "" for d, _ in head]
            self._apply_threads(self.threads)          # 正式口径：单线程
            self._score_pairs(query, texts, batch_size=1)
            times.append((time.perf_counter() - t0) * 1000.0)
        self._apply_threads(self.score_threads or self.threads)
        times.sort()
        return round(times[len(times) // 2], 3)

    def __repr__(self) -> str:
        return (f"Reranker({self.model_path or self.model_name}, backend={self.backend}, "
                f"depth={self.depth}, topn={self.topn}, device={self.device})")
