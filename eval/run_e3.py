#!/usr/bin/env python
"""E3：切块策略消融 —— 负责人：B（检索）

    python eval/run_e3.py --split full                 # 契约三档：整段 / 256 / 512+64
    python eval/run_e3.py --split smoke --grid extended  # 扩展五档（含句级、128、重叠）

要复现报告里的数字请钉住 int8 权重（``--encoder auto`` 会优先 fp32，指标会高出约 0.4 个点）::

    python eval/run_e3.py --split full --model-path models/bge-small-zh-v1.5-onnx/model_int8.onnx

做法与口径：

1. 按策略把 3507 个段落切成检索单元（chunk），重建稠密索引与 BM25 倒排；
2. 检索得到 chunk 级排名，再**压平成段落级排名**（同一段落只保留最好名次），
   与 E2 的 ``Recall@5`` 完全同口径，可以直接横向比；
3. 额外记录 ``chunk_recall@5``（命中 chunk 的金标段落占比）、
   ``context_chars@5``（top-5 拼进上下文的平均字数，下游阅读器的成本）、
   ``chunks_per_doc``（索引膨胀倍数）与延迟。

产出::

    results/raw/E3_chunk_para.json / E3_chunk_256.json / E3_chunk_512.json
    results/raw/E3_summary.json
    results/tab/E3_chunk.csv
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
from retrieval.bge_encoder import BgeEncoder, FaissIndex  # noqa: E402
from retrieval.bm25 import BM25  # noqa: E402
from retrieval.chunking import ChunkStrategy, build_chunks, chunks_to_doc_ranking  # noqa: E402
from retrieval.rrf import fuse  # noqa: E402
from retrieval.tokenization import build_tokenizer  # noqa: E402

# 契约三档（run_exp.MATRIX["E3"]）+ 扩展档
STRATEGIES = {
    "paragraph": (ChunkStrategy("whole"), "E3_chunk_para"),
    "256": (ChunkStrategy("fixed", size=256, overlap=0), "E3_chunk_256"),
    "512": (ChunkStrategy("fixed", size=512, overlap=64), "E3_chunk_512"),
    "sentence": (ChunkStrategy("sentence"), "E3_chunk_sentence"),
    "128": (ChunkStrategy("fixed", size=128, overlap=0), "E3_chunk_128"),
    "256o64": (ChunkStrategy("fixed", size=256, overlap=64), "E3_chunk_256o64"),
    "512o128": (ChunkStrategy("fixed", size=512, overlap=128), "E3_chunk_512o128"),
}
CONTRACT = ["paragraph", "256", "512"]
EXTENDED = ["paragraph", "sentence", "128", "256", "256o64", "512", "512o128"]


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


def _latency_probe(encoder, index, bm25, questions, retriever, alpha, depth, topk, fuse_k,
                   n_probe: int = 20) -> float:
    """单查询端到端（编码 + 检索 + 融合 + 压平），取中位数，单位 ms。"""
    sample = list(questions)[: min(n_probe, len(questions))]
    times = []
    for _ in range(3):
        for q in sample:
            t0 = time.perf_counter()
            q_emb = encoder.encode_query(q)
            if retriever == "bm25":
                bm25.topk_ids(q, topk=depth)
            elif retriever == "dense":
                index.search_ids(q_emb, depth)
            else:
                sparse = bm25.topk_ids(q, topk=depth)
                dense = index.search_ids(q_emb, depth)
                fuse([[(d, 0.0) for d in sparse], [(d, 0.0) for d in dense]],
                     weights=[1.0 - alpha, alpha], k=fuse_k)[:topk]
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def main() -> int:
    ap = argparse.ArgumentParser(description="E3: 切块策略消融")
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="full", choices=["smoke", "full"])
    ap.add_argument("--grid", default="contract", choices=["contract", "extended"])
    ap.add_argument("--strategies", default="", help="逗号分隔，覆盖 --grid，如 paragraph,256")
    ap.add_argument("--retriever", default="dense", choices=["dense", "rrf", "hybrid", "bm25"],
                    help="契约默认 dense；rrf/hybrid 同义（接力棒里的 --retriever dense|bm25|hybrid）")
    ap.add_argument("--alpha", type=float, default=None, help="rrf 的稠密权重，缺省取 config")
    ap.add_argument("--also-hybrid", action="store_true", help="每档额外跑一遍 α=0.5 的混合检索")
    ap.add_argument("--encoder", default="auto", choices=["auto", "onnx", "st", "hash"])
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--limit-queries", type=int, default=0)
    args = ap.parse_args()

    add_repo_to_path()
    cfg = load_config(args.config)
    fuse_k = int(cfg.get("retrieval", {}).get("fusion", {}).get("k", 60))
    alpha = args.alpha if args.alpha is not None else float(
        cfg.get("retrieval", {}).get("fusion", {}).get("alpha", 0.5))
    depth = args.depth or int(cfg.get("retrieval", {}).get("fusion", {}).get("dense_depth", 100))
    topk = int(cfg.get("eval", {}).get("topk", 5))
    boot_n = int(cfg.get("eval", {}).get("bootstrap_n", 1000))
    boot_alpha = float(cfg.get("eval", {}).get("alpha", 0.05))
    seed = int(cfg.get("project", {}).get("seed", 42))
    tok_name = cfg.get("retrieval", {}).get("sparse", {}).get("tokenizer", "jieba")
    k1 = float(cfg.get("retrieval", {}).get("sparse", {}).get("k1", 1.5))
    b = float(cfg.get("retrieval", {}).get("sparse", {}).get("b", 0.75))

    names = [s.strip() for s in args.strategies.split(",") if s.strip()] if args.strategies else (
        CONTRACT if args.grid == "contract" else EXTENDED)
    for n in names:
        if n not in STRATEGIES:
            raise SystemExit(f"未知切块策略 {n!r}，可选 {sorted(STRATEGIES)}")

    log = Logger("E3_run", cfg)
    log(f"[E3] split={args.split} strategies={names} retriever={args.retriever} "
        f"alpha={alpha} depth={depth} topk={topk} encoder={args.encoder}")
    t_all = time.time()

    docs, queries = load_split(args.split, cfg)
    if args.limit_queries:
        queries = queries[: args.limit_queries]
    questions = [q["question"] for q in queries]
    golds = [q["gold_doc_id"] for q in queries]
    qids = [q["qid"] for q in queries]
    doc_ids = [d["doc_id"] for d in docs]
    doc_texts = [d["text"] for d in docs]
    log(f"[data] n_docs={len(docs)} n_questions={len(questions)}")

    encoder = _encoder_from_cfg(cfg, args.encoder, args.model_path)
    t0 = time.time()
    q_emb = encoder.encode_queries(questions, show_progress=True)
    log(f"[dense] 查询编码 {q_emb.shape}，{time.time() - t0:.1f}s | encoder={encoder.model_name} "
        f"backend={encoder._engine}")

    mode = "rrf" if args.retriever == "hybrid" else args.retriever   # hybrid 与 rrf 同义
    if args.also_hybrid and mode == "dense":
        mode = "dense"

    rows = []
    for name in names:
        strategy, run_id = STRATEGIES[name]
        t0 = time.time()
        chunks = build_chunks(doc_texts, strategy)
        chunk_texts = [c.text for c in chunks]
        index = FaissIndex().build_from_encoder(encoder, [{"doc_id": c.chunk_id, "text": c.text}
                                                         for c in chunks],
                                                tag=f"chunk_{name}")
        bm25 = BM25(k1=k1, b=b, tokenizer=build_tokenizer(None if tok_name in ("auto", "jieba") else tok_name))
        bm25.fit([{"doc_id": str(c.chunk_id), "text": c.text} for c in chunks])
        build_s = time.time() - t0
        log(f"[{name}] chunks={len(chunks)} (×{len(chunks)/len(docs):.2f}/段) "
            f"avg {sum(len(t) for t in chunk_texts)/max(1,len(chunk_texts)):.0f} 字，"
            f"建索引 {build_s:.1f}s")

        dense_rows = index.search_ids_batch(q_emb, topk=depth)
        variants = [(mode, alpha if mode == "rrf" else None)]
        if args.also_hybrid and mode != "rrf":
            variants.append(("rrf", alpha))

        for retriever, a in variants:
            t0 = time.time()
            doc_rankings, chunk_hits, ctx_chars = [], [], []
            for i, q in enumerate(questions):
                if retriever == "bm25":
                    fused = [(int(d), 0.0) for d in bm25.topk_ids(q, topk=depth)]
                elif retriever == "dense":
                    fused = [(int(d), 0.0) for d in dense_rows[i]]
                else:
                    sparse = bm25.topk_ids(q, topk=depth)
                    fused = fuse(
                        [[(d, 0.0) for d in sparse], [(d, 0.0) for d in dense_rows[i]]],
                        weights=[1.0 - a, a], k=fuse_k,
                    )
                top = fused[:max(topk, 5)]
                doc_rank = chunks_to_doc_ranking([cid for cid, _ in top], chunks)
                doc_rankings.append([doc_ids[d] for d in doc_rank][:topk])
                gold = golds[i]
                chunk_hits.append(1.0 if any(doc_ids[chunks[cid].doc_id] == gold for cid, _ in top[:5]) else 0.0)
                ctx_chars.append(sum(len(chunks[cid].text) for cid, _ in top[:5]))
            query_s = time.time() - t0

            met = RE.metrics_dict(doc_rankings, golds, ks=(1, 5, 10, 20), ndcg_k=5)
            ci = RE.recall_ci(doc_rankings, golds, 5, boot_n, boot_alpha, seed)
            met["latency_ms"] = round(_latency_probe(encoder, index, bm25, questions, retriever,
                                                     a, depth, topk, fuse_k), 3)
            met["recall@5_ci"] = ci
            met["chunk_recall@5"] = round(float(sum(chunk_hits) / max(1, len(chunk_hits))), 6)
            met["n_chunks"] = len(chunks)
            met["chunks_per_doc"] = round(len(chunks) / max(1, len(docs)), 3)
            met["avg_chunk_chars"] = round(sum(len(t) for t in chunk_texts) / max(1, len(chunk_texts)), 1)
            met["context_chars@5"] = round(sum(ctx_chars) / max(1, len(ctx_chars)), 1)

            suffix = "" if retriever == "dense" and not args.also_hybrid else (
                "" if retriever == mode else f"_{retriever}alpha{a:.1f}")
            rid = f"{run_id}{suffix}"
            path = write_run(
                rid,
                config={"retriever": retriever, "alpha": a, "chunk": name,
                        "chunk_size": getattr(strategy, "size", None),
                        "chunk_overlap": getattr(strategy, "overlap", None),
                        "rrf_k": fuse_k if retriever == "rrf" else None,
                        "depth": depth, "topk": topk, "rerank": False,
                        "encoder": args.encoder, "model": encoder.model_name,
                        "model_file": Path(str(encoder.model_file)).name,
                        "index_backend": index.backend,
                        "tokenizer": str(bm25.tokenizer)},
                metrics=met,
                per_question=RE.per_question_records(qids, doc_rankings, golds),
                meta={"split": args.split, "n_questions": len(questions),
                      "dataset": {"n_docs": len(docs), "n_chunks": len(chunks)},
                      "index_build_s": round(build_s, 2)},
                cfg=cfg,
            )
            log(RE.format_line(f"[{rid}]", met, ci)
                + f" chunkR@5={met['chunk_recall@5']:.4f} ctx={met['context_chars@5']:.0f}字 "
                  f"lat={met['latency_ms']:.2f}ms → {path.name}")
            rows.append({"run_id": rid, "strategy": name, "retriever": retriever, "alpha": a, **met})

    dump_json({"experiment": "E3", "split": args.split, "rows": rows,
               "config": {"depth": depth, "topk": topk, "rrf_k": fuse_k,
                          "encoder": args.encoder, "grid": args.grid}},
              path_of(cfg, "results_raw") / "E3_summary.json")
    table = [{"run_id": r["run_id"], "strategy": r["strategy"],
              "recall@5": r["recall@5"], "recall@5_lo": r["recall@5_ci"][0],
              "recall@5_hi": r["recall@5_ci"][1], "mrr": r["mrr"], "ndcg@5": r["ndcg@5"],
              "chunk_recall@5": r["chunk_recall@5"], "chunks_per_doc": r["chunks_per_doc"],
              "avg_chunk_chars": r["avg_chunk_chars"], "context_chars@5": r["context_chars@5"],
              "latency_ms": r["latency_ms"]} for r in rows]
    path = write_table("E3_chunk", table, cfg)
    log(f"[done] 汇总 → results/raw/E3_summary.json ；表 → {path.name} ；总耗时 {time.time() - t_all:.1f}s")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
