"""预热重排打分缓存：E4/E5 共用同一份 pool，避免重复跑交叉编码器。

    python scripts/_warm_rerank.py --split smoke --limit-queries 100 --pool 100

产物：``.cache/rerank/pool_*.json``（被 .gitignore 挡住，换机器要重新生成）。
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import c_pipeline as CP
from eval.io_utils import load_config
from rerank.cross_encoder import Reranker

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="smoke")
ap.add_argument("--limit-queries", type=int, default=100)
ap.add_argument("--pool", type=int, default=100)
ap.add_argument("--alpha", type=float, default=0.2)
ap.add_argument("--model-path", default="models/bge-small-zh-v1.5-onnx/model_int8.onnx")
ap.add_argument("--rerank-model", default=None)
ap.add_argument("--rerank-backend", default="auto")
args = ap.parse_args()

cfg = load_config()
t0 = time.time()
bundle = CP.build_candidates(cfg, split=args.split, alpha=args.alpha, depth=args.pool,
                             model_path=args.model_path, limit=args.limit_queries)
print(f"[warm] 候选就绪 {len(bundle)} 题，{time.time() - t0:.0f}s", flush=True)

reranker = Reranker(depth=args.pool, topn=5, device="cpu", threads=1, score_threads=18,
                    model_path=args.rerank_model, backend=args.rerank_backend
                    ).set_corpus(bundle.docs)
pooled = CP.rerank_pool(bundle, reranker, pool=args.pool, use_cache=True)
print(f"[warm] 完成，{len(pooled)} 题，总耗时 {time.time() - t0:.0f}s")
