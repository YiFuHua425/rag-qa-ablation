"""BGE 稠密检索 + FAISS 索引 —— 负责人：B（检索）

状态：[READY]

两个类，签名冻结（见 INTERFACE.md）：

* :class:`BgeEncoder` —— 把文本编码成 **L2 归一化**的稠密向量
* :class:`FaissIndex` —— 内积索引（``IndexFlatIP``），等价余弦相似度

三种后端（``backend=``）：

============  =========================================================  ==================
后端          依赖                                                        场景
============  =========================================================  ==================
``onnx``      onnxruntime + tokenizer.json（不装 torch，约 11 MB）        本机实测走这条
``st``        sentence-transformers + torch                              有 GPU 的机器
``hash``      无                                                          离线冒烟，数字不可用
============  =========================================================  ==================

要点（也是报告第 4 章「实现过程」的素材）：

* 模型名读 ``config.yaml``，不在代码里写死；``_model`` 懒加载，
  ``--dry-run`` 不会触发下载。
* doc 侧一次性编码并落盘缓存（``.cache/emb/*.npz``，``.gitignore`` 已忽略），
  缓存键含「模型文件 + max_length + 归一化开关 + 语料内容摘要」，任何一项变了都会失效。
* 编码完 ``assert`` 每行范数 ≈ 1（踩坑记录：忘了归一化时 Recall@5 掉 4.8 个点）。
* faiss 未安装时自动退化为 numpy 暴力内积（3507 篇、512 维，单查询约 0.3 ms），
  两条路径的 top-k 结果在自检里断言一致。
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = PKG_ROOT / ".cache" / "emb"
_MODEL_CACHE: Dict[str, object] = {}

STUB = False


# ---------------------------------------------------------------- 工具
def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return (x / norm).astype(np.float32)


def _resolve_onnx_dir(model_name: str, model_path: Optional[str]) -> Path:
    if model_path:
        return Path(model_path)
    env = os.environ.get("BGE_MODEL_PATH")
    if env:
        return Path(env)
    short = model_name.split("/")[-1]
    for cand in (f"{short}-onnx", short, f"{short}-onnx-int8"):
        p = PKG_ROOT / "models" / cand
        if p.exists():
            return p
    return PKG_ROOT / "models" / f"{short}-onnx"


# ---------------------------------------------------------------- 编码器
class BgeEncoder:
    """把文本编码成 L2 归一化向量。签名冻结，见 INTERFACE.md。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5",
                 query_prefix: str = "", normalize: bool = True,
                 batch_size: int = 64, device: str | None = None,
                 model_path: str | None = None, backend: str = "auto",
                 max_length: int = 512, threads: int = 0) -> None:
        self.model_name = model_name
        self.query_prefix = query_prefix or ""
        self.normalize = normalize
        self.batch_size = batch_size
        self.device = device
        self.model_path = model_path
        self.backend = backend
        self.max_length = max_length
        self.threads = threads
        self._model = None                      # 懒加载：--dry-run 不许触发模型下载
        self._engine: str | None = None

    # ------------------------------------------------------------ 懒加载
    def _pick_backend(self) -> str:
        if self.backend != "auto":
            return self.backend
        path = _resolve_onnx_dir(self.model_name, self.model_path)
        # model_path 既可以是目录（自动挑 model.onnx → model_int8.onnx），
        # 也可以直接给某个 .onnx 文件，用来固定量化口径（如强制 int8 复现正式数字）
        if path.suffix.lower() == ".onnx" and path.is_file():
            return "onnx"
        if path.is_dir() and any(path.glob("*.onnx")):
            return "onnx"
        try:  # pragma: no cover - 只有装了 torch 的机器会走到
            import sentence_transformers  # noqa: F401

            return "st"
        except Exception:
            pass
        raise RuntimeError(
            "找不到可用的稠密编码器：\n"
            f"  1) 没有 ONNX 模型：{path}（放 model.onnx 或 model_int8.onnx）\n"
            "  2) 也没装 sentence-transformers/torch\n"
            "解法：跑 `python scripts/download_model.py` 下 ONNX 权重，"
            "或 `pip install sentence-transformers`，或离线冒烟时用 --encoder hash"
        )

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        engine = self._pick_backend()
        key = (f"{engine}:{self.model_name}:{self.model_path}:{self.max_length}:"
               f"{self.threads}:{self.normalize}")
        if key in _MODEL_CACHE:
            self._model, self._engine = _MODEL_CACHE[key], engine
            return self._model

        if engine == "onnx":
            from .onnx_encoder import ONNXBGEEncoder

            path = _resolve_onnx_dir(self.model_name, self.model_path)
            model = ONNXBGEEncoder(
                path,
                max_length=self.max_length,
                batch_size=self.batch_size,
                use_query_instruction=False,   # 前缀由本类统一控制（走 config.yaml）
                threads=self.threads,
                normalize=self.normalize,
            )
        elif engine == "st":  # pragma: no cover
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self.model_name, device=self.device)
        elif engine == "hash":
            from .hash_encoder import HashEncoder

            model = HashEncoder()
        else:
            raise ValueError(f"未知后端 {engine!r}")

        _MODEL_CACHE[key] = model
        self._model, self._engine = model, engine
        return model

    # ------------------------------------------------------------ 属性
    @property
    def dim(self) -> int:
        return int(getattr(self._ensure_model(), "dim"))

    @property
    def model_file(self) -> str:
        return str(getattr(self._ensure_model(), "model_file", self.model_name))

    # ------------------------------------------------------------ 编码
    def _encode_texts(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        model = self._ensure_model()
        texts = [str(t) for t in texts]
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        emb = model.encode(texts, batch_size=self.batch_size, show_progress=show_progress)
        emb = np.asarray(emb, dtype=np.float32)
        if self.normalize:
            emb = _l2_normalize(emb)
        if show_progress and self.normalize:
            norms = np.linalg.norm(emb, axis=1)
            assert np.allclose(norms, 1.0, atol=1e-3), f"向量未归一化：min={norms.min():.4f}"
        return emb

    def encode_docs(self, docs: List[dict], show_progress: bool = True) -> np.ndarray:
        """docs: [{"doc_id","text"}]（也接受纯字符串列表）-> (N, d)。"""
        return self.encode_doc_texts([d["text"] if isinstance(d, dict) else d for d in docs],
                                     show_progress=show_progress)

    def encode_doc_texts(self, texts: Sequence[str], show_progress: bool = True) -> np.ndarray:
        return self._encode_texts(texts, show_progress=show_progress)

    def encode_query(self, query: str) -> np.ndarray:
        """-> (d,)，已 L2 归一化。"""
        return self.encode_queries([query])[0]

    def encode_queries(self, queries: Sequence[str], show_progress: bool = False) -> np.ndarray:
        prefix = self.query_prefix
        texts = [f"{prefix}{q}" for q in queries] if prefix else list(queries)
        return self._encode_texts(texts, show_progress=show_progress)

    def __repr__(self) -> str:
        engine = self._engine or self.backend
        return f"BgeEncoder({self.model_name}, backend={engine}, dim={getattr(self, '_dim_hint', '?')})"


# ---------------------------------------------------------------- 索引
class FaissIndex:
    """FAISS 内积索引（缺 faiss 时自动退化 numpy）。签名冻结，见 INTERFACE.md。"""

    def __init__(self, dim: int | None = None, index_type: str = "IndexFlatIP",
                 use_faiss: bool = True) -> None:
        if index_type not in ("IndexFlatIP",):
            raise ValueError("当前只实现 IndexFlatIP（内积）；向量已归一化，等价余弦")
        self.dim = dim
        self.index_type = index_type
        self.use_faiss = use_faiss
        self.index = None
        self.doc_ids: List[str] = []
        self._embs: Optional[np.ndarray] = None
        self._backend: str = "numpy"
        self._id_to_row: Dict[str, int] = {}

    # ------------------------------------------------------------ 建索引
    def build(self, doc_ids: List[str], embs) -> "FaissIndex":
        embs = np.asarray(embs, dtype=np.float32)
        if embs.ndim != 2:
            raise ValueError(f"embs 必须是二维矩阵，拿到 {embs.shape}")
        if len(doc_ids) != embs.shape[0]:
            raise ValueError(f"doc_ids({len(doc_ids)}) 与向量行数({embs.shape[0]}) 不一致")
        if len(set(doc_ids)) != len(doc_ids):
            raise ValueError("doc_ids 有重复，RRF 融合会静默出错")
        self.doc_ids = list(doc_ids)
        self._id_to_row = {d: i for i, d in enumerate(self.doc_ids)}
        self.dim = int(embs.shape[1])
        self._embs = embs

        if self.use_faiss:
            try:  # pragma: no cover - 本机未装 faiss
                import faiss

                self.index = faiss.IndexFlatIP(self.dim)
                self.index.add(embs)
                self._backend = "faiss"
                return self
            except Exception:
                self.index = None
                self._backend = "numpy"
        return self

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def size(self) -> int:
        return len(self.doc_ids)

    # ------------------------------------------------------------ 检索
    def _require_built(self) -> None:
        if not self.doc_ids or self._embs is None:
            raise RuntimeError("FaissIndex.search() 前必须先 build()")

    def search_batch(self, q_embs, topk: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """批量检索，返回 (scores, rows)，形状均为 (n_q, k)。"""
        self._require_built()
        q = np.atleast_2d(np.asarray(q_embs, dtype=np.float32))
        k = min(int(topk), len(self.doc_ids))
        if self._backend == "faiss" and self.index is not None:  # pragma: no cover
            scores, rows = self.index.search(q, k)
            return scores, rows
        scores_all = q @ self._embs.T                       # (n_q, N)
        if k >= scores_all.shape[1]:
            rows = np.argsort(-scores_all, axis=1, kind="stable")
        else:
            part = np.argpartition(-scores_all, kth=k - 1, axis=1)[:, :k]
            part_scores = np.take_along_axis(scores_all, part, axis=1)
            order = np.argsort(-part_scores, axis=1, kind="stable")
            rows = np.take_along_axis(part, order, axis=1)
        scores = np.take_along_axis(scores_all, rows, axis=1)
        return scores, rows

    def search(self, q_emb, topk: int = 100) -> List[Tuple[str, float]]:
        scores, rows = self.search_batch(q_emb, topk)
        return [(self.doc_ids[int(r)], float(s)) for r, s in zip(rows[0], scores[0])]

    def search_ids(self, q_emb, topk: int = 100) -> List[str]:
        return [d for d, _ in self.search(q_emb, topk)]

    def search_ids_batch(self, q_embs, topk: int = 100) -> List[List[str]]:
        _, rows = self.search_batch(q_embs, topk)
        return [[self.doc_ids[int(r)] for r in row] for row in rows]

    # ------------------------------------------------------------ 缓存
    @staticmethod
    def cache_path(cache_dir: Path | str, tag: str, stamp: str, n: int) -> Path:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"emb_{tag}_{stamp}_n{n}.npz".replace(" ", "_")

    def build_from_encoder(self, encoder: "BgeEncoder", docs: List[dict], *,
                           cache_dir: Path | str | None = DEFAULT_CACHE_DIR,
                           tag: str = "docs", show_progress: bool = True,
                           doc_ids: Optional[List[str]] = None) -> "FaissIndex":
        """编码 + 建索引，doc 向量落盘复用（接口契约要求的缓存）。"""
        texts = [d["text"] if isinstance(d, dict) else d for d in docs]
        ids = doc_ids if doc_ids is not None else [
            d["doc_id"] if isinstance(d, dict) else str(i) for i, d in enumerate(docs)
        ]
        emb = None
        path = None
        if cache_dir is not None:
            # 缓存键里必须含**语料内容摘要**：doc_id 编号方案一变（例如第 1 棒 D 的数据
            # 从 p_00000 起改成 p_00001 起），条目数 n 仍旧相同，只按 n 命中会把
            # 旧向量错配到新 doc_id 上，指标会静默偏移。
            h = hashlib.sha1()
            for i, t in zip(ids, texts):
                h.update(str(i).encode("utf-8"))
                h.update(b"\x00")
                h.update(str(t).encode("utf-8"))
                h.update(b"\x01")
            digest = h.hexdigest()[:12]
            stamp = (f"{encoder._engine or encoder.backend}-{Path(str(encoder.model_file)).name}"
                     f"-{encoder.max_length}-{'norm' if encoder.normalize else 'raw'}-{digest}")
            path = self.cache_path(cache_dir, tag, stamp, len(texts))
            if path.exists():
                try:
                    with np.load(path) as z:
                        emb = z["embeddings"]
                        cached = str(z["digest"]) if "digest" in z.files else ""
                    if emb.shape[0] != len(texts) or cached != digest:
                        emb = None
                    else:
                        print(f"[enc] 命中缓存 {path.name} {emb.shape}", flush=True)
                except Exception:
                    emb = None
        if emb is None:
            t0 = time.time()
            emb = encoder.encode_doc_texts(texts, show_progress=show_progress)
            print(f"[enc] 文档编码完成 {emb.shape}，用时 {time.time() - t0:.1f}s", flush=True)
            if path is not None:
                np.savez_compressed(path, embeddings=emb, digest=np.array(digest))
        return self.build(ids, emb)

    def __repr__(self) -> str:
        return f"FaissIndex(size={self.size}, dim={self.dim}, backend={self._backend})"
