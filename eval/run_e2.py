#!/usr/bin/env python
"""E2：BM25 / 稠密 / RRF 融合的 α 扫描 —— 负责人：B（检索）

    python eval/run_e2.py --split full            # 七档 α + 两条单路基线
    python eval/run_e2.py --split smoke --limit-queries 100
    python eval/run_e2.py --encoder hash --split smoke   # 零模型的冒烟

**固定量化口径**：目录里同时有 model.onnx（fp32）与 model_int8.onnx 时，``--encoder auto``
会优先挑 fp32，数字会与报告的 int8 表略有出入。要复现报告里的数字，显式钉住权重：

    python eval/run_e2.py --split full --model-path models/bge-small-zh-v1.5-onnx/model_int8.onnx

产出（格式见 INTERFACE.md 第 3 节）::

    results/raw/E2_bm25.json
    results/raw/E2_rrf_alpha{0.0,0.2,0.4,0.5,0.6,0.8,1.0}.json
    results/raw/E2_summary.json          ← 画图用的汇总
    results/tab/E2_alpha.csv             ← 报告里的表
    results/log/E2_*.log

run_id 与 ``eval/run_exp.py`` 的 E2 配置矩阵**逐字对齐**，D 直接拿去画图。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import retrieval_eval as RE  # noqa: E402
from eval.io_utils import (Logger, add_repo_to_path, dump_json, load_config, load_split,
                           path_of, write_run, write_table)  # noqa: E402
from retrieval.bge_encoder import DEFAULT_CACHE_DIR, BgeEncoder, FaissIndex  # noqa: E402
from retrieval.bm25 import BM25  # noqa: E402
from retrieval.rrf import fuse  # noqa: E402
from retrieval.tokenization import build_tokenizer  # noqa: E402

DEFAULT_ALPHAS = "0.0,0.2,0.4,0.5,0.6,0.8,1.0"   # 与 run_exp.MATRIX["E2"] 一致


def _bm25_from_cfg(cfg: dict) -> BM25:
    k1 = float(cfg.get("retrieval", {}).get("sparse", {}).get("k1", 1.5))
    b = float(cfg.get("retrieval", {}).get("sparse", {}).get("b", 0.75))
    tok_name = cfg.get("retrieval", {}).get("sparse", {}).get("tokenizer", "jieba")
    tok = build_tokenizer(None if tok_name in ("auto", "jieba") else tok_name)
    return BM25(k1=k1, b=b, tokenizer=tok)


def _encoder_from_cfg(cfg: dict, backend: str, model_path: str | None) -> BgeEncoder:
    dense = cfg.get("retrieval", {}).get("dense", {})
    return BgeEncoder(
        model_name=dense.get("model", "BAAI/bge-small-zh-v1.5"),
        query_prefix=dense.get("query_prefix", ""),
        normalize=bool(dense.get("normalize", True)),
        batch_size=int(dense.get("encode_batch", 64)),
        model_path=model_path,
        backend=backend,
    )


def _latency_probe(bm25: BM25, index: FaissIndex, encoder: BgeEncoder,
                   questions, alpha: float, depth: int, topk: int, k: int) -> float:
    """单查询端到端（BM25 打分 + 稠密内积 + RRF），取中位数，单位 ms。"""
    sample = questions[: min(30, len(questions))]
    times = []
    for _ in range(3):
        for q in sample:
            t0 = time.perf_counter()
            sparse = bm25.topk_ids(q, topk)
            dense = index.search_ids(encoder.encode_query(q), depth)
            fuse([[(d, 0.0) for d in sparse], [(d, 0.0) for d in dense]],
                 weights=[1.0 - alpha, alpha], k=k)[:topk]
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def main() -> int:
    ap = argparse.ArgumentParser(description="E2: BM25/稠密/RRF α 扫描")
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="full", choices=["smoke", "full"])
    ap.add_argument("--encoder", default="auto", choices=["auto", "onnx", "st", "hash"])
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--alphas", default=DEFAULT_ALPHAS)
    ap.add_argument("--depth", type=int, default=None, help="两路候选深度，缺省取 fusion.dense_depth")
    ap.add_argument("--sparse-topk", type=int, default=None, help="缺省取 retrieval.sparse.topk")
    ap.add_argument("--limit-queries", type=int, default=0)
    ap.add_argument("--no-cache", action="store_true", help="不复用 doc 向量缓存")
    ap.add_argument("--with-k1b-sweep", action="store_true", help="额外扫 BM25 的 k1/b（改进实验）")
    args = ap.parse_args()

    add_repo_to_path()
    cfg = load_config(args.config)
    fuse_k = int(cfg.get("retrieval", {}).get("fusion", {}).get("k", 60))
    depth = args.depth or int(cfg.get("retrieval", {}).get("fusion", {}).get("dense_depth", 100))
    sparse_topk = args.sparse_topk or int(cfg.get("retrieval", {}).get("sparse", {}).get("topk", 100))
    topk = int(cfg.get("eval", {}).get("topk", 5))
    ks = (1, 5, 10, 20)
    boot_n = int(cfg.get("eval", {}).get("bootstrap_n", 1000))
    boot_alpha = float(cfg.get("eval", {}).get("alpha", 0.05))
    seed = int(cfg.get("project", {}).get("seed", 42))
    alphas = [float(x) for x in str(args.alphas).split(",") if x.strip()]

    log = Logger("E2_run", cfg)
    log(f"[E2] split={args.split} encoder={args.encoder} alphas={alphas} depth={depth} "
        f"sparse_topk={sparse_topk} fuse_k={fuse_k}")
    t_all = time.time()

    docs, queries = load_split(args.split, cfg)
    if args.limit_queries:
        queries = queries[: args.limit_queries]
    questions = [q["question"] for q in queries]
    golds = [q["gold_doc_id"] for q in queries]
    qids = [q["qid"] for q in queries]
    log(f"[data] n_docs={len(docs)} n_questions={len(questions)} "
        f"(avg doc {sum(len(d['text']) for d in docs) / len(docs):.0f} 字)")

    # ---------------- 稀疏路 ----------------
    t0 = time.time()
    bm25 = _bm25_from_cfg(cfg)
    bm25.fit(docs)
    log(f"[bm25] {bm25} | tokenizer={bm25.tokenizer} | fit {time.time() - t0:.1f}s")
    t0 = time.time()
    sparse_runs = bm25.rank_ids_batch(questions, topk=sparse_topk, progress=False)
    log(f"[bm25] 打分完成 {len(sparse_runs)} 题，{time.time() - t0:.1f}s")

    # ---------------- 稠密路 ----------------
    encoder = _encoder_from_cfg(cfg, args.encoder, args.model_path)
    index = FaissIndex()
    index.build_from_encoder(
        encoder, docs,
        cache_dir=DEFAULT_CACHE_DIR if not args.no_cache else None,
        tag="docs",
    )
    log(f"[dense] {index} | encoder={encoder.model_name} backend={encoder._engine}")
    t0 = time.time()
    q_emb = encoder.encode_queries(questions, show_progress=True)
    log(f"[dense] 查询编码 {q_emb.shape}，{time.time() - t0:.1f}s")
    dense_runs = index.search_ids_batch(q_emb, topk=depth)
    log(f"[dense] 检索完成，depth={depth}")

    # ---------------- 单路基线 ----------------
    summary = []

    def _record(run_id: str, rankings, extra_config: dict, tag: str):
        met = RE.metrics_dict(rankings, golds, ks=ks, ndcg_k=5)
        ci = RE.recall_ci(rankings, golds, 5, boot_n, boot_alpha, seed)
        lat = _latency_probe(bm25, index, encoder, questions, extra_config.get("alpha", 0.0),
                             depth, topk, fuse_k)
        met["latency_ms"] = round(lat, 3)
        met["recall@5_ci"] = ci
        config = {
            "retriever": extra_config.get("retriever"),
            "alpha": extra_config.get("alpha"),
            "fusion": "rrf" if extra_config.get("retriever") == "rrf" else None,
            "rrf_k": fuse_k if extra_config.get("retriever") == "rrf" else None,
            "depth": depth,
            "sparse_topk": sparse_topk,
            "topk": topk,
            "rerank": False,
            "encoder": args.encoder,
            "model": encoder.model_name,
            "model_file": Path(str(encoder.model_file)).name,
            "index_backend": index.backend,
            "tokenizer": str(bm25.tokenizer),
            "k1": bm25.k1,
            "b": bm25.b,
        }
        path = write_run(
            run_id,
            config=config,
            metrics={k: v for k, v in met.items()},
            per_question=RE.per_question_records(qids, rankings, golds),
            meta={"split": args.split, "n_questions": len(questions), "dataset": {
                "n_docs": len(docs),
                "avg_doc_chars": round(sum(len(d["text"]) for d in docs) / len(docs), 1),
            }},
            cfg=cfg,
        )
        log(RE.format_line(f"[{tag}]", met, ci) + f"  lat={met['latency_ms']:.2f}ms → {path.name}")
        summary.append({"run_id": run_id, "tag": tag, **{k: met[k] for k in met},
                        "alpha": extra_config.get("alpha"), "retriever": extra_config.get("retriever")})

    top_rank = lambda runs: [r[:topk] for r in runs]
    _record("E2_bm25", top_rank(sparse_runs), {"retriever": "bm25"}, "E2_bm25")
    _record("E2_dense", top_rank(dense_runs), {"retriever": "dense", "alpha": 1.0}, "E2_dense")

    # ---------------- α 扫描 ----------------
    for alpha in alphas:
        t0 = time.time()
        rankings = []
        for s_run, d_run in zip(sparse_runs, dense_runs):
            fused = fuse(
                [[(d, 0.0) for d in s_run], [(d, 0.0) for d in d_run]],
                weights=[1.0 - alpha, alpha], k=fuse_k,
            )
            rankings.append([d for d, _ in fused[:topk]])
        run_id = f"E2_rrf_alpha{alpha:.1f}"
        _record(run_id, rankings, {"retriever": "rrf", "alpha": alpha}, f"α={alpha:.1f}")
        log(f"       融合 {len(rankings)} 题，{time.time() - t0:.1f}s")

    # ---------------- 可选：BM25 参数扫描（改进实验） ----------------
    if args.with_k1b_sweep:
        log("[sweep] BM25 k1/b 网格（只看 Recall@5 / MRR）")
        best = None
        for k1 in (0.9, 1.2, 1.5, 2.0):
            for b in (0.3, 0.5, 0.75, 0.9):
                probe = BM25(k1=k1, b=b, tokenizer=bm25.tokenizer).fit(docs)
                runs = probe.rank_ids_batch(questions, topk=sparse_topk)
                met = RE.metrics_dict([r[:topk] for r in runs], golds, ks=(1, 5), ndcg_k=5)
                summary.append({"run_id": f"E2_bm25_k1{k1}_b{b}", "tag": "k1/b",
                                "alpha": None, "retriever": "bm25",
                                **{k: met[k] for k in met if k.startswith(("recall", "mrr", "ndcg"))},
                                "k1": k1, "b": b})
                log(f"        k1={k1:<4} b={b:<5} R@5={met['recall@5']:.4f} MRR={met['mrr']:.4f}")
                if best is None or met["recall@5"] > best[2]:
                    best = (k1, b, met["recall@5"])
        log(f"[sweep] 最优 k1={best[0]} b={best[1]} R@5={best[2]:.4f}")

    summary.sort(key=lambda r: (r.get("alpha") is None, r.get("alpha") or 0.0))
    dump_json({"experiment": "E2", "split": args.split, "rows": summary,
               "config": {"depth": depth, "sparse_topk": sparse_topk, "rrf_k": fuse_k,
                          "topk": topk, "encoder": args.encoder, "alphas": alphas}},
              path_of(cfg, "results_raw") / "E2_summary.json")
    table = [
        {"run_id": r["run_id"], "alpha": r.get("alpha"), "recall@1": r.get("recall@1"),
         "recall@5": r.get("recall@5"), "recall@5_lo": (r.get("recall@5_ci") or [None, None])[0],
         "recall@5_hi": (r.get("recall@5_ci") or [None, None])[1],
         "mrr": r.get("mrr"), "ndcg@5": r.get("ndcg@5"), "latency_ms": r.get("latency_ms")}
        for r in summary if r.get("recall@5") is not None
    ]
    path = write_table("E2_alpha", table, cfg)
    log(f"[done] 汇总 → results/raw/E2_summary.json ；表 → {path.name} ；总耗时 {time.time() - t_all:.1f}s")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
