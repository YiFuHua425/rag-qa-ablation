"""faiss 后端 vs numpy 暴力内积：一致性 + 延迟对比（B 的补充实验）。

背景：本组最初因为 pip/沙箱问题没装上 faiss，正式实验走的是 numpy 内积。
后来 faiss-cpu 从清华镜像装上了，于是补做这一组对照，回答两个问题：

1. 两个后端的 top-k 是否完全一致（一致性）？
2. faiss 在 3507×512 这个规模上是否真的更快（延迟）？

输出：``results/tab/E2_backend.csv`` + ``results/raw/E2_backend.json``

用法：
    python eval/run_backend_check.py [--split full] [--topk 100] [--n-latency 300]
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import eval.io_utils as IU  # noqa: E402
from eval import retrieval_eval as RE  # noqa: E402
from retrieval import BgeEncoder, BM25, FaissIndex, fuse_ids  # noqa: E402
from retrieval.rrf import alpha_weights  # noqa: E402


def timed(fn, repeat: int = 3) -> float:
    best = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best.append((time.perf_counter() - t0) * 1000)
    return statistics.median(best)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="full", choices=["smoke", "full"])
    ap.add_argument("--topk", type=int, default=100, help="两路候选深度（= E2 的 depth）")
    ap.add_argument("--n-latency", type=int, default=300, help="延迟测量的查询条数")
    ap.add_argument("--alpha", type=float, default=0.2, help="融合权重（用 E2 最优值）")
    ap.add_argument("--model-path", default=None,
                    help="钉死权重文件，保证与正式口径一致（默认 int8，见 docs/B_检索模块说明.md 5.5）")
    ap.add_argument("--repeat", type=int, default=5, help="延迟重复轮数，取每轮中位数")
    args = ap.parse_args()

    # 与正式口径一致：默认钉 int8。目录里同时放了 model.onnx（fp32）与 model_int8.onnx 时，
    # 后端 auto 探测会优先 fp32，单路稠密会高约 0.44 个点，与本表对不上。
    if args.model_path is None:
        pinned = Path(__file__).resolve().parents[1] / "models/bge-small-zh-v1.5-onnx/model_int8.onnx"
        if pinned.exists():
            args.model_path = str(pinned)
            print(f"[pin] 未指定 --model-path，按正式口径钉 int8：{pinned.name}", flush=True)

    cfg = IU.load_config()
    docs, queries = IU.load_split(args.split, cfg)
    questions = [q["question"] for q in queries]
    golds = [q["gold_doc_id"] for q in queries]
    print(f"[data] docs={len(docs)} queries={len(questions)}", flush=True)

    encoder = BgeEncoder(model_path=args.model_path)
    print(f"[enc] backend={encoder.backend} model={Path(str(encoder.model_file)).name}", flush=True)

    idx_faiss = FaissIndex(use_faiss=True).build_from_encoder(encoder, docs, tag="docs")
    idx_numpy = FaissIndex(use_faiss=False).build(idx_faiss.doc_ids, idx_faiss._embs)
    print(f"[idx] faiss={idx_faiss.backend} numpy={idx_numpy.backend} size={idx_faiss.size}", flush=True)

    t0 = time.time()
    q_embs = np.asarray(encoder.encode_queries(questions, show_progress=False), dtype=np.float32)
    print(f"[enc] 查询编码 {q_embs.shape}，{time.time() - t0:.1f}s", flush=True)

    # ---- 一致性：top-k 文档 id 列表是否逐位相同 ----
    ids_faiss = idx_faiss.search_ids_batch(q_embs, args.topk)
    ids_numpy = idx_numpy.search_ids_batch(q_embs, args.topk)
    same_pos = sum(a == b for a, b in zip(ids_faiss, ids_numpy)) / len(ids_faiss)
    same_set = sum(set(a) == set(b) for a, b in zip(ids_faiss, ids_numpy)) / len(ids_faiss)
    print(f"[consistency] top-{args.topk} 逐位相同={same_pos:.4%} 集合相同={same_set:.4%}", flush=True)

    # ---- 延迟：单查询 top-k ----
    probe = q_embs[: min(args.n_latency, len(q_embs))]

    def run_faiss():
        for q in probe:
            idx_faiss.search_batch(q.reshape(1, -1), args.topk)

    def run_numpy():
        for q in probe:
            idx_numpy.search_batch(q.reshape(1, -1), args.topk)

    run_faiss(); run_numpy()                      # 预热，避免首轮把 numpy/faiss 初始化算进去
    lat_faiss = timed(run_faiss, repeat=args.repeat) / len(probe)
    lat_numpy = timed(run_numpy, repeat=args.repeat) / len(probe)
    print(f"[latency] faiss={lat_faiss:.3f} ms/query  numpy={lat_numpy:.3f} ms/query", flush=True)

    # ---- 端到端指标：单路 + RRF（验证后端不影响结论）----
    bm25 = BM25(k1=float(cfg["retrieval"]["sparse"]["k1"]), b=float(cfg["retrieval"]["sparse"]["b"]))
    bm25.fit(docs)
    dense_runs = [ids_faiss[i] for i in range(len(ids_faiss))]
    dense_runs_np = [ids_numpy[i] for i in range(len(ids_numpy))]
    sparse_runs = [bm25.search(q, topk=args.topk) for q in questions]
    sparse_ids = [[d for d, _ in run] for run in sparse_runs]
    fusion = cfg.get("retrieval", {}).get("fusion", {})
    fuse_k = int(fusion.get("k", 60))          # 超参一律走 config.yaml
    w = alpha_weights(args.alpha)

    rows = []
    for name, d_runs, lat in (("faiss", dense_runs, lat_faiss), ("numpy", dense_runs_np, lat_numpy)):
        dense_only = RE.metrics_dict(d_runs, golds)
        fused = []
        for i in range(len(questions)):
            fused.append(fuse_ids([sparse_ids[i], d_runs[i]], weights=w, k=fuse_k))
        m = RE.metrics_dict(fused, golds)
        rows.append({
            "backend": name,
            "identical_topk": round(same_pos if name == "faiss" else 1.0, 6),
            "dense_recall@5": round(dense_only["recall@5"], 6),
            "dense_mrr": round(dense_only["mrr"], 6),
            "rrf_alpha": args.alpha,
            "rrf_k": fuse_k,
            "rrf_recall@5": round(m["recall@5"], 6),
            "rrf_mrr": round(m["mrr"], 6),
            "rrf_ndcg@5": round(m["ndcg@5"], 6),
            "latency_ms": round(lat, 3),
        })
        print(f"[{name}] dense R@5={dense_only['recall@5']:.4f}  "
              f"RRF R@5={m['recall@5']:.4f} MRR={m['mrr']:.4f}  {lat:.3f} ms", flush=True)

    IU.write_table("E2_backend", rows, cfg)
    IU.dump_json({
        "split": args.split,
        "topk": args.topk,
        "n_queries": len(questions),
        "consistency": {"same_position": same_pos, "same_set": same_set},
        "rows": rows,
        "model": Path(str(encoder.model_file)).name,
    }, IU.path_of(cfg, "results_raw") / "E2_backend.json")
    print("[done] results/tab/E2_backend.csv", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
