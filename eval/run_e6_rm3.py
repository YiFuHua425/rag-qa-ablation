#!/usr/bin/env python
"""E6：RM3 伪相关反馈（查询改写）消融 —— 负责人：B（加分项）

    python eval/run_e6_rm3.py --split full

RM3（Relevance Model 3，Lavrenko & Croft 2001）三步：

1. 用原始查询检索 top-``fb_docs``（默认 10）篇文档，作为"伪相关文档集"；
2. 估计相关模型 ``P(w|R) ∝ Σ_d P(w|θ_d)·P(d|q)``，其中 ``P(w|θ_d)=tf(w,d)/|d|``，
   ``P(d|q)=softmax(BM25 分数)``；取概率最高的 ``fb_terms``（默认 20）个词；
3. 与原查询线性插值 ``P(w|q') = (1-γ)·P(w|q) + γ·P(w|R)``（γ=0.5），
   用加权查询重新检索。

口径：**只改查询、不动任何索引与超参**，baseline 与 RM3 的唯一差异是查询本身。
产出 ``results/raw/E6_*.json`` + ``results/tab/E6_rm3.csv`` + ``results/raw/E6_summary.json``。
"""

from __future__ import annotations

import argparse
import math
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


def rm3_expand(bm25: BM25, query: str, *, fb_docs: int = 10, fb_terms: int = 20,
               gamma: float = 0.5) -> tuple[dict, list[str]]:
    """返回 ``({term: weight}, 扩展词列表)``。"""
    ranked = bm25.search(query, topk=fb_docs)
    if not ranked:
        return {t: 1.0 for t in bm25.query_terms(query)}, []
    # P(d|q)：BM25 分数做 softmax（除以均值温度，避免过度平滑）
    scores = [s for _, s in ranked]
    mean = sum(scores) / len(scores) or 1.0
    exps = [math.exp((s - max(scores)) / max(mean, 1e-6)) for s in scores]
    z = sum(exps) or 1.0
    p_dq = [e / z for e in exps]

    rel_model: dict[str, float] = {}
    id2row = {d: i for i, d in enumerate(bm25.doc_ids)}
    rows = [id2row[d] for d, _ in ranked]
    for row, p in zip(rows, p_dq):
        length = int(bm25.doc_len[row]) or 1
        for term, tf in bm25.vocab_terms_of(row).items():
            rel_model[term] = rel_model.get(term, 0.0) + p * tf / length

    top = sorted(rel_model.items(), key=lambda kv: -kv[1])[:fb_terms]
    total = sum(v for _, v in top) or 1.0
    q_terms = bm25.query_terms(query)
    if not q_terms:
        return {}, [t for t, _ in top[:5]]
    weights = {t: (1.0 - gamma) / len(q_terms) for t in q_terms}
    for term, v in top:
        weights[term] = weights.get(term, 0.0) + gamma * v / total
    expanded = [t for t, _ in top if t not in q_terms][:5]
    return weights, expanded


def main() -> int:
    ap = argparse.ArgumentParser(description="E6: RM3 伪相关反馈")
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="full", choices=["smoke", "full"])
    ap.add_argument("--fb-docs", type=int, default=10)
    ap.add_argument("--fb-terms", type=int, default=20)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--encoder", default="auto", choices=["auto", "onnx", "st", "hash"])
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--alpha", type=float, default=None, help="混合检索的稠密权重")
    ap.add_argument("--limit-queries", type=int, default=0)
    args = ap.parse_args()

    add_repo_to_path()
    cfg = load_config(args.config)
    topk = int(cfg.get("eval", {}).get("topk", 5))
    fuse_k = int(cfg.get("retrieval", {}).get("fusion", {}).get("k", 60))
    depth = int(cfg.get("retrieval", {}).get("fusion", {}).get("dense_depth", 100))
    alpha = args.alpha if args.alpha is not None else float(
        cfg.get("retrieval", {}).get("fusion", {}).get("alpha", 0.5))
    boot_n = int(cfg.get("eval", {}).get("bootstrap_n", 1000))
    boot_alpha = float(cfg.get("eval", {}).get("alpha", 0.05))
    seed = int(cfg.get("project", {}).get("seed", 42))
    tok_name = cfg.get("retrieval", {}).get("sparse", {}).get("tokenizer", "jieba")

    log = Logger("E6_run", cfg)
    log(f"[E6] split={args.split} fb_docs={args.fb_docs} fb_terms={args.fb_terms} gamma={args.gamma} "
        f"alpha={alpha} depth={depth}")
    t_all = time.time()

    docs, queries = load_split(args.split, cfg)
    if args.limit_queries:
        queries = queries[: args.limit_queries]
    questions = [q["question"] for q in queries]
    golds = [q["gold_doc_id"] for q in queries]
    qids = [q["qid"] for q in queries]

    bm25 = BM25(k1=float(cfg["retrieval"]["sparse"].get("k1", 1.5)),
                b=float(cfg["retrieval"]["sparse"].get("b", 0.75)),
                tokenizer=build_tokenizer(None if tok_name in ("auto", "jieba") else tok_name))
    bm25.fit(docs)
    log(f"[bm25] {bm25}")

    dense = cfg.get("retrieval", {}).get("dense", {})
    encoder = BgeEncoder(model_name=dense.get("model", "BAAI/bge-small-zh-v1.5"),
                         query_prefix=dense.get("query_prefix", ""),
                         normalize=bool(dense.get("normalize", True)),
                         batch_size=int(dense.get("encode_batch", 64)),
                         model_path=args.model_path, backend=args.encoder)
    index = FaissIndex().build_from_encoder(encoder, docs, cache_dir=DEFAULT_CACHE_DIR, tag="docs")
    q_emb = encoder.encode_queries(questions)
    dense_runs = index.search_ids_batch(q_emb, topk=depth)
    log(f"[dense] {index} | {encoder.model_name}")

    base_sparse = [bm25.topk_ids(q, topk=depth) for q in questions]
    rows_summary = []

    def _eval_and_write(run_id: str, label: str, sparse_runs, mode: str) -> dict:
        if mode == "bm25":
            rankings = [r[:topk] for r in sparse_runs]
        else:
            rankings = []
            for s_run, d_run in zip(sparse_runs, dense_runs):
                fused = fuse([[(d, 0.0) for d in s_run], [(d, 0.0) for d in d_run]],
                             weights=[1.0 - alpha, alpha], k=fuse_k)
                rankings.append([d for d, _ in fused[:topk]])
        met = RE.metrics_dict(rankings, golds, ks=(1, 5, 10, 20), ndcg_k=5)
        ci = RE.recall_ci(rankings, golds, 5, boot_n, boot_alpha, seed)
        met["recall@5_ci"] = ci
        write_run(run_id,
                  config={"retriever": mode, "alpha": None if mode == "bm25" else alpha,
                          "query_rewrite": None if "baseline" in run_id else "rm3",
                          "fb_docs": args.fb_docs, "fb_terms": args.fb_terms,
                          "gamma": args.gamma, "depth": depth, "topk": topk, "rerank": False,
                          "encoder": args.encoder, "model": encoder.model_name,
                          "model_file": Path(str(encoder.model_file)).name,
                          "index_backend": index.backend},
                  metrics=met,
                  per_question=RE.per_question_records(qids, rankings, golds),
                  meta={"split": args.split, "n_questions": len(questions)},
                  cfg=cfg)
        log(RE.format_line(f"[{run_id}]", met, ci) + f"  ({label})")
        row = {"run_id": run_id, "label": label, "mode": mode, **met}
        rows_summary.append(row)
        return row

    b_base = _eval_and_write("E6_baseline_bm25", "BM25 基线", base_sparse, "bm25")
    b_rrf = _eval_and_write("E6_baseline_rrf", "RRF 基线", base_sparse, "rrf")

    t0 = time.time()
    rm3_sparse, examples = [], []
    for i, q in enumerate(questions):
        weights, expanded = rm3_expand(bm25, q, fb_docs=args.fb_docs,
                                       fb_terms=args.fb_terms, gamma=args.gamma)
        rm3_sparse.append([d for d, _ in bm25.search_weighted(weights, topk=depth)])
        if i < 5:
            examples.append({"qid": qids[i], "query": q, "expanded": expanded})
    log(f"[rm3] 查询改写完成 {len(questions)} 题，{time.time() - t0:.1f}s；示例：")
    for ex in examples:
        log(f"        {ex['qid']} {ex['query']}  →  +{ex['expanded']}")

    r_base = _eval_and_write("E6_rm3_bm25", "RM3 + BM25", rm3_sparse, "bm25")
    r_rrf = _eval_and_write("E6_rm3_rrf", "RM3 + RRF", rm3_sparse, "rrf")

    def _delta(a: dict, b: dict) -> str:
        d = (b["recall@5"] - a["recall@5"]) * 100
        return f"R@5 {a['recall@5']:.4f} → {b['recall@5']:.4f} ({d:+.2f} pt)"

    log(f"[E6] BM25 路：{_delta(b_base, r_base)}")
    log(f"[E6] RRF  路：{_delta(b_rrf, r_rrf)}")

    dump_json({"experiment": "E6", "split": args.split, "rows": rows_summary,
               "params": {"fb_docs": args.fb_docs, "fb_terms": args.fb_terms,
                          "gamma": args.gamma, "alpha": alpha, "depth": depth},
               "examples": examples},
              path_of(cfg, "results_raw") / "E6_summary.json")
    table = [{"run_id": r["run_id"], "label": r["label"], "recall@1": r["recall@1"],
              "recall@5": r["recall@5"], "recall@5_lo": r["recall@5_ci"][0],
              "recall@5_hi": r["recall@5_ci"][1], "mrr": r["mrr"], "ndcg@5": r["ndcg@5"]}
             for r in rows_summary]
    path = write_table("E6_rm3", table, cfg)
    log(f"[done] 表 → {path.name} ；总耗时 {time.time() - t_all:.1f}s")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
