"""ONNX 版 BGE 编码器：不装 torch，只要 onnxruntime（约 11 MB）。

    from retrieval.onnx_encoder import ONNXBGEEncoder
    enc = ONNXBGEEncoder("models/bge-small-zh-v1.5-onnx")
    v = enc.encode_queries(["中国的首都是哪里"])

输出取 ``last_hidden_state[:, 0, :]``（CLS 池化）+ L2 归一化，与
``sentence-transformers`` 的 ``bge-small-zh-v1.5`` 约定一致。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .bert_tokenizer import BertWordPieceTokenizer

QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return (x / norm).astype(np.float32)


class ONNXBGEEncoder:
    def __init__(
        self,
        model_path: str | Path,
        tokenizer_path: Optional[str | Path] = None,
        max_length: int = 512,
        batch_size: int = 32,
        use_query_instruction: bool = True,
        threads: int = 0,
        normalize: bool = True,
    ) -> None:
        import onnxruntime as ort  # 延迟导入，测试不依赖

        model_path = Path(model_path)
        if model_path.is_dir():
            candidates = ["model.onnx", "model_int8.onnx", "model_quantized.onnx"]
            found = [model_path / c for c in candidates if (model_path / c).exists()]
            if not found:
                raise FileNotFoundError(f"{model_path} 下没有 onnx 模型（找过 {candidates}）")
            model_file = found[0]
            tokenizer_path = tokenizer_path or model_path / "tokenizer.json"
        else:
            model_file = model_path
            tokenizer_path = tokenizer_path or model_path.parent / "tokenizer.json"

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            opts.intra_op_num_threads = int(threads)
        self.session = ort.InferenceSession(str(model_file), sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.output_name = self.session.get_outputs()[0].name
        self.model_file = str(model_file)
        self.tokenizer = BertWordPieceTokenizer.from_tokenizer_json(tokenizer_path, max_length=max_length)
        self.tokenizer_path = str(tokenizer_path)
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_query_instruction = use_query_instruction
        self.normalize = normalize

    def __repr__(self) -> str:
        return f"ONNXBGEEncoder({Path(self.model_file).name}, dim={self.dim})"

    @property
    def dim(self) -> int:
        for out in self.session.get_outputs():
            shape = out.shape
            if len(shape) == 3 and isinstance(shape[-1], int):
                return int(shape[-1])
        return 512

    # ---------------- 内部 ----------------
    def _run_batch(self, texts: Sequence[str]) -> np.ndarray:
        input_ids, attention = self.tokenizer.batch_encode(list(texts), max_length=self.max_length)
        feeds = {
            "input_ids": np.asarray(input_ids, dtype=np.int64),
            "attention_mask": np.asarray(attention, dtype=np.int64),
        }
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.zeros_like(feeds["input_ids"])
        outputs = self.session.run([self.output_name], feeds)
        hidden = outputs[0]
        if hidden.ndim != 3:
            raise RuntimeError(f"模型输出维度异常: {hidden.shape}")
        emb = hidden[:, 0, :].astype(np.float32)
        return _l2_normalize(emb) if self.normalize else emb

    def encode(self, texts: Sequence[str], batch_size: Optional[int] = None, show_progress: bool = False) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        bs = batch_size or self.batch_size
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for start in range(0, len(texts), bs):
            chunk = texts[start : start + bs]
            out[start : start + len(chunk)] = self._run_batch(chunk)
            if show_progress and (start // bs) % 20 == 0:
                print(f"  [onnx] {start + len(chunk)}/{len(texts)}", flush=True)
        return out

    def encode_documents(self, texts: Sequence[str], **kwargs) -> np.ndarray:
        return self.encode(texts, **kwargs)

    def encode_queries(self, texts: Sequence[str], **kwargs) -> np.ndarray:
        if self.use_query_instruction:
            texts = [QUERY_INSTRUCTION + t for t in texts]
        return self.encode(texts, **kwargs)
