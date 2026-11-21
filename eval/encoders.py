"""编码器选择 + 文档向量缓存（E2/E3 共用）。

三种后端：

* ``onnx``：onnxruntime + 纯 Python WordPiece，本机唯一可用的真实 BGE；
* ``bge`` ：transformers + torch（装了 torch 的机器上更快、更"标准"）；
* ``hash``：无语义的伪编码器，仅供离线冒烟，数值不得进报告。

缓存文件名里带上「后端 + 模型文件名 + 语料规模 + 模型 mtime」，
换了量化版本或换了语料会自动失效，避免拿到过期向量。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from retrieval.dense import make_encoder

from .common import DATA_DIR, MODELS_DIR

ONNX_DIR = "bge-small-zh-v1.5-onnx"
TORCH_DIR = "bge-small-zh-v1.5"


def resolve_model_path(kind: str, model_path: Optional[str]) -> Path:
    if model_path:
        return Path(model_path)
    return MODELS_DIR / (ONNX_DIR if kind == "onnx" else TORCH_DIR)


def build_encoder(kind: str, model_path: Optional[str], batch_size: int = 32):
    path = resolve_model_path(kind, model_path)
    kwargs = {"batch_size": batch_size}
    if kind == "onnx":
        kwargs["threads"] = 0  # 交给 onnxruntime 自己决定线程数
    return make_encoder(kind, path, **kwargs)


def _model_stamp(kind: str, model_path: Path) -> str:
    if model_path.is_dir():
        files = sorted(p for p in model_path.glob("*.onnx")) or sorted(model_path.glob("*.safetensors"))
        if files:
            f = files[0]
            return f"{f.name}-{int(f.stat().st_mtime)}"
        return "dir"
    return f"{model_path.name}-{int(model_path.stat().st_mtime)}" if model_path.exists() else "missing"


def doc_embeddings(
    encoder,
    docs: List[str],
    kind: str,
    model_path: Path,
    cache_dir: Optional[Path] = None,
    tag: str = "",
) -> np.ndarray:
    """有缓存读缓存，没有就编码并存盘。"""
    cache_dir = cache_dir or (DATA_DIR / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = f"emb_{kind}_{tag or 'docs'}_{_model_stamp(kind, model_path)}_n{len(docs)}.npz".replace(" ", "_")
    path = cache_dir / name
    if path.exists():
        emb = np.load(path)["embeddings"]
        if emb.shape[0] == len(docs):
            print(f"[enc] 命中缓存 {path.name} {emb.shape}", flush=True)
            return emb
        print("[enc] 缓存规模不符，重新编码", flush=True)
    import time

    t0 = time.time()
    emb = encoder.encode_documents(docs, show_progress=True)
    print(f"[enc] 文档编码完成 {emb.shape}，用时 {time.time()-t0:.1f}s", flush=True)
    np.savez_compressed(path, embeddings=emb)
    return emb
