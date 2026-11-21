"""离线冒烟用的伪编码器：不加载任何模型，纯哈希把文本映射成稠密向量。

用途只有一个：在没装 onnxruntime / torch，或者模型还没下载完的机器上，
把 E2/E3 的**代码路径**跑通（接口、缓存、融合、落盘、画图）。
它的向量**没有任何语义**，Recall 会明显低，数字绝对不许写进报告。
"""

from __future__ import annotations

import zlib
from typing import Sequence

import numpy as np


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return (x / norm).astype(np.float32)


class HashEncoder:
    """字符 n-gram → 哈希桶累加 → L2 归一化。确定性、零依赖、单条微秒级。"""

    def __init__(self, dim: int = 512, ngram: int = 2, seed: int = 42) -> None:
        self.dim = int(dim)
        self.ngram = int(ngram)
        self.seed = int(seed)
        self.model_name = f"hash-{dim}-{ngram}"
        self.model_file = f"hash-{dim}-{ngram}"
        self.batch_size = 64
        self.max_length = 512

    def _encode_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        text = (text or "").strip()
        if not text:
            return vec
        for n in range(1, self.ngram + 1):
            for i in range(max(0, len(text) - n + 1)):
                gram = text[i : i + n]
                h = zlib.crc32(f"{self.seed}:{gram}".encode("utf-8"))
                vec[h % self.dim] += 1.0
        return vec

    def encode(self, texts: Sequence[str], batch_size: int | None = None,
               show_progress: bool = False) -> np.ndarray:
        out = np.stack([self._encode_one(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)
        return _l2_normalize(out)

    encode_documents = encode

    def encode_queries(self, texts: Sequence[str], **kwargs) -> np.ndarray:
        return self.encode(texts, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover
        return f"HashEncoder(dim={self.dim}, ngram={self.ngram})"
