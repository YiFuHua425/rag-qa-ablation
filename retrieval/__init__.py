"""检索模块包 —— 负责人：B（检索）

导出（供 pipeline / 实验脚本使用）::

    from retrieval.bm25 import BM25
    from retrieval.bge_encoder import BgeEncoder, FaissIndex
    from retrieval.rrf import fuse
    from retrieval.chunking import ChunkStrategy, build_chunks, chunks_to_doc_ranking

状态：

* ``bm25.py``        [READY] 手写 BM25（倒排 + numpy 向量化打分）
* ``bge_encoder.py`` [READY] BGE 编码（onnx / sentence-transformers）+ 内积索引
* ``rrf.py``         [READY] 手写 RRF
* ``chunking.py``    [READY] E3 切块策略（整段 / 句 / 定长 + 重叠）
* ``tokenization.py``    中文分词（jieba，缺失时退化为字 + 二元组）
* ``onnx_encoder.py`` / ``bert_tokenizer.py``  ONNX 推理与纯 Python WordPiece
* ``hash_encoder.py``    离线冒烟用的伪编码器（数字不可用于报告）
"""

from .bge_encoder import BgeEncoder, FaissIndex
from .bm25 import BM25
from .chunking import Chunk, ChunkStrategy, build_chunks, chunks_to_doc_ranking, strategy_grid
from .rrf import alpha_weights, fuse, fuse_ids

__all__ = [
    "BM25",
    "BgeEncoder",
    "FaissIndex",
    "fuse",
    "fuse_ids",
    "alpha_weights",
    "Chunk",
    "ChunkStrategy",
    "build_chunks",
    "chunks_to_doc_ranking",
    "strategy_grid",
]
