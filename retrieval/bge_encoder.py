"""BGE 稠密检索 + FAISS 索引 —— 负责人：B（检索）

状态：[STUB] 空实现，等 B 替换为真实现。

要点：
  * 向量必须 **L2 归一化**，索引用内积（IndexFlatIP）等价余弦相似度
  * doc 侧一次性编码并落盘缓存（`data/emb_docs.npy` + `data/doc_ids.json`），
    否则每次实验都要重编码，一晚上跑不完
  * 模型名走 config.yaml，不许写死在代码里

实现清单（B 逐条打勾）：
  [ ] encode_docs 分批（encode_batch），不要一次吃进全部文本
  [ ] 编码完 assert 每行 norm ≈ 1.0
  [ ] 索引建好后缓存；第二次运行直接 load
  [ ] search 返回 [(doc_id, float(score))]，score = 内积（余弦）
  [ ] 与 BM25 联调时确认 doc_id 集合完全一致（否则 RRF 会静默漏召回）
"""

from __future__ import annotations

import warnings

STUB = True
_WARNED = False


def _warn_once() -> None:
    global _WARNED
    if not _WARNED:
        warnings.warn(
            "retrieval/bge_encoder.py 仍是 STUB，返回空结果；B 请替换为真实现。",
            RuntimeWarning,
            stacklevel=2,
        )
        _WARNED = True


class BgeEncoder:
    """把文本编码成 L2 归一化向量。签名冻结，见 INTERFACE.md。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5",
                 query_prefix: str = "", normalize: bool = True,
                 batch_size: int = 64, device: str | None = None) -> None:
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.normalize = normalize
        self.batch_size = batch_size
        self.device = device
        self._model = None  # 懒加载：--dry-run 不许触发模型下载

    def _ensure_model(self):
        # TODO(B): 懒加载 SentenceTransformer
        # from sentence_transformers import SentenceTransformer
        # if self._model is None:
        #     self._model = SentenceTransformer(self.model_name, device=self.device)
        _warn_once()
        raise NotImplementedError(
            "BgeEncoder._ensure_model() 待 B 实现（真实运行需要 torch + sentence-transformers）"
        )

    def encode_docs(self, docs: list[dict]):
        """-> np.ndarray (N, d)，已 L2 归一化。"""
        self._ensure_model()
        raise NotImplementedError

    def encode_query(self, query: str):
        """-> np.ndarray (d,)，已 L2 归一化。"""
        self._ensure_model()
        raise NotImplementedError


class FaissIndex:
    """FAISS 内积索引。签名冻结，见 INTERFACE.md。"""

    def __init__(self, dim: int | None = None, index_type: str = "IndexFlatIP") -> None:
        self.dim = dim
        self.index_type = index_type
        self.index = None
        self.doc_ids: list[str] = []

    def build(self, doc_ids: list[str], embs) -> None:
        _warn_once()
        # TODO(B): faiss.IndexFlatIP(dim); index.add(embs); self.doc_ids = doc_ids
        self.doc_ids = list(doc_ids)

    def search(self, q_emb, topk: int = 100) -> list[tuple[str, float]]:
        _warn_once()
        if not self.doc_ids:
            raise RuntimeError("FaissIndex.search() 前必须先 build()")
        # TODO(B)
        return []