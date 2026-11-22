#!/usr/bin/env python
"""E4：重排与候选深度（精度–延迟 Pareto）—— 负责人：C

    python eval/run_e4.py --split smoke --limit-queries 200            # 正式口径（200 题）
    python eval/run_e4.py --split smoke --no-reader --limit-queries 20 # 只验检索侧，秒级
    python eval/run_e4.py --split smoke --depths 20,50 --no-plot

实验设计（接力棒第 3、4 棒）：

===================  ==========================================================
arm                 含义
===================  ==========================================================
``E4_rerank_off``    RRF(α) 直接取前 5 —— 基线，也是"重排到底值不值"的分母
``E4_rerank_d20``    只对前 20 个候选重排
``E4_rerank_d50``    前 50（config 默认）
``E4_rerank_d100``   前 100（等于整个候选池）
===================  ==========================================================

**只动 depth，不动检索**：三个 arm 共用同一份候选（``.cache/cand/`` 缓存），
这样差异只能来自重排本身。检索配置固定为 E2 的最优 α=0.2（B 的结论）。

延迟口径（config.yaml 的 ``latency`` 段）：**CPU 单线程、batch=1、重复 3 次取中位数**。
指标计算用 batch=16 的吞吐模式，两者分开跑，报告里写清楚。

产出::

    results/raw/E4_rerank_{off,d20,d50,d100}.json
    results/raw/E4_summary.json
    results/tab/E4_rerank.csv
    results/fig/E4_pareto.png
    results/log/E4_run.log
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from eval import c_pipeline as CP  # noqa: E402
from eval import retrieval_eval as RE  # noqa: E402
from eval.io_utils import (Logger, dump_json, load_config, path_of,  # noqa: E402
                           write_run, write_table)


def _cfg_path(cfg: dict, key: str, default: str) -> Path:
    return ROOT / str((cfg.get("paths", {}) or {}).get(key, default))


def main() -> int:
    ap = argparse.ArgumentParser(description="E4: 重排与候选深度")
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--limit-queries", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.2, help="固定检索配置（E2 最优）")
    ap.add_argument("--pool", type=int, default=100, help="候选池深度（要 ≥ 最大 depth）")
    ap.add_argument("--depths", default="20,50,100")
    ap.add_argument("--topn", type=int, default=5)
    ap.add_argument("--order", default="natural")
    ap.add_argument("--max-chars", type=int, default=2000)
    ap.add_argument("--reader-model", default=None)
    ap.add_argument("--rerank-model", default=None)
    ap.add_argument("--rerank-backend", default="auto", choices=["auto", "hf", "onnx", "lexical"],
                    help="lexical 只用于无模型的环境验证链路，数字不可进报告")
    ap.add_argument("--encoder", default="auto", choices=["auto", "onnx", "st", "hash"])
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--latency-queries", type=int, default=3)
    ap.add_argument("--latency-repeat", type=int, default=2)
    ap.add_argument("--no-rerank", action="store_true", help="只跑检索基线（不做任何重排）")
    ap.add_argument("--no-reader", action="store_true", help="只跑检索/重排，不算 F1（快）")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    depths = [int(x) for x in str(args.depths).split(",") if x.strip()]
    if max(depths) > args.pool:
        raise SystemExit(f"--pool({args.pool}) 必须 ≥ 最大 depth({max(depths)})")

    seed = int((cfg.get("project", {}) or {}).get("seed", 42))
    boot_n = int((cfg.get("eval", {}) or {}).get("bootstrap_n", 1000))
    boot_alpha = float((cfg.get("eval", {}) or {}).get("alpha", 0.05))
    lat_cfg = cfg.get("latency", {}) or {}
    threads = int(lat_cfg.get("threads", 1))
    repeats = int(lat_cfg.get("repeat", 3))
    rerank_cfg = cfg.get("rerank", {}) or {}
    reader_cfg = cfg.get("reader", {}) or {}
    ctx_cfg = cfg.get("context", {}) or {}
    # 算指标用多线程（快），报延迟用 config 里的单线程（口径干净）——两件事分开
    score_threads = max(1, (os.cpu_count() or 3) - 2)

    log = Logger("E4_run", cfg)
    log(f"[E4] split={args.split} alpha={args.alpha} pool={args.pool} depths={depths} "
        f"topn={args.topn} reader={'off' if args.no_reader else 'on'}")

    t_all = time.time()
    bundle = CP.build_candidates(cfg, split=args.split, alpha=args.alpha, depth=args.pool,
                                 encoder_backend=args.encoder, model_path=args.model_path,
                                 limit=args.limit_queries, use_cache=not args.no_cache)
    log(f"[data] n_docs={len(bundle.docs)} n_questions={len(bundle)} | 检索延迟 "
        f"{CP.retrieval_latency(bundle):.2f} ms/query | {bundle.info}")

    # ---------------- 重排与阅读器 ----------------
    from rerank.cross_encoder import Reranker

    reranker = Reranker(
        model_name=rerank_cfg.get("model", "BAAI/bge-reranker-base"),
        depth=max(depths), topn=args.topn,
        device=str(lat_cfg.get("device", "cpu")),   # 延迟口径：CPU 单线程
        max_length=int(rerank_cfg.get("max_length", 512)),
        model_path=args.rerank_model, threads=threads, backend=args.rerank_backend,
        score_threads=score_threads,
    ).set_corpus(bundle.docs)

    reader = None
    reader_latency = 0.0
    if not args.no_reader:
        from reader.extractive_qa import ExtractiveReader

        reader = ExtractiveReader(
            model_name=reader_cfg.get("model", "uer/roberta-base-chinese-extractive-qa"),
            max_length=int(reader_cfg.get("max_length", 512)),
            device=str(reader_cfg.get("device", "cpu")),
            batch_size=int(reader_cfg.get("batch_size", 16)),
            null_score=float(reader_cfg.get("null_score", -100.0)),
            model_path=args.reader_model, threads=threads, score_threads=score_threads,
        )
        probe = [(bundle.questions[i], bundle.doc_map[bundle.golds[i]])
                 for i in range(min(args.latency_queries, len(bundle)))]
        reader_latency = reader.benchmark_latency(probe, repeat=repeats)
        log(f"[reader] 单条延迟 {reader_latency:.1f} ms/query（batch=1 × {repeats} 轮中位数）")

    # ---------------- 逐 arm 跑 ----------------
    base_lat = CP.retrieval_latency(bundle)
    summary = []
    print("-" * 96)
    print(f"{'run_id':18s}{'R@1':>7s}{'R@5':>7s}{'MRR':>7s}{'nDCG':>7s}"
          f"{'F1':>7s}{'kept':>7s}{'rerank':>9s}{'total':>8s}  状态")
    print("-" * 96)

    arms = [("E4_rerank_off", None)] + [(f"E4_rerank_d{d}", d) for d in depths]
    # 三个 depth 共用一次打分（见 c_pipeline.rerank_pool 的等价性说明）
    pooled = CP.rerank_pool(bundle, reranker, pool=args.pool, use_cache=not args.no_cache) \
        if not args.no_rerank else []
    for run_id, depth in arms:
        t0 = time.time()
        if depth is None:
            ranked = [[d for d, _ in cand[: args.topn]] for cand in bundle.candidates]
            rerank_ms = 0.0
        else:
            ranked = CP.topk_from_pool(pooled, depth, args.topn)
            reranker.depth = depth                     # benchmark 按当前 depth 拼候选
            probe_idx = range(min(args.latency_queries, len(bundle)))
            lat = [reranker.benchmark_latency(bundle.questions[i], bundle.candidates[i],
                                              repeat=args.latency_repeat) for i in probe_idx]
            rerank_ms = CP.median(lat)

        met = CP.rank_metrics(ranked, bundle.golds, ndcg_k=args.topn)
        met["recall@5_ci"] = RE.recall_ci(ranked, bundle.golds, 5, boot_n, boot_alpha, seed)
        met["retrieval_ms"] = base_lat
        met["rerank_ms"] = rerank_ms
        met["reader_ms"] = reader_latency
        met["latency_ms"] = round(base_lat + rerank_ms + reader_latency, 3)

        reading = None
        if reader is not None:
            reading = CP.read_answers(reader, bundle, ranked,
                                      order=str(ctx_cfg.get("order", args.order)),
                                      max_chars=int(args.max_chars),
                                      progress=f"{run_id}")
            met.update(CP.summarize_reading(reading))
            met["f1_ci"] = CP.bootstrap_f1(reading["rows"], boot_n, boot_alpha, seed)
        else:
            met.update({"f1": None, "em": None, "gold_kept_rate": None})

        rows = RE.per_question_records(bundle.qids, ranked, bundle.golds, top_show=args.topn)
        if reading is not None:
            for row, rd in zip(rows, reading["rows"]):
                row.update({"pred": rd["pred"], "f1": rd["f1"],
                            "gold_used": rd["gold_used"]})

        config = {
            "retriever": "rrf", "alpha": args.alpha, "pool": args.pool,
            "rerank": depth is not None, "depth": depth, "topn": args.topn,
            "order": str(ctx_cfg.get("order", args.order)), "max_chars": int(args.max_chars),
            "reader": "off" if reader is None else str(reader_cfg.get("model")),
            "rerank_model": str(rerank_cfg.get("model")),
            "rerank_backend": reranker.engine,
            "rerank_depth_ms": rerank_ms, "reader_ms": reader_latency,
            "retrieval_ms": base_lat, "encoder": args.encoder,
        }
        path = write_run(run_id, config=config, metrics=met, per_question=rows,
                         meta={"split": args.split, "n_questions": len(bundle),
                               "dataset": {"n_docs": len(bundle.docs)},
                               "latency": {"threads": threads, "repeat": repeats,
                                           "batch": 1, "device": lat_cfg.get("device", "cpu")}},
                         cfg=cfg)
        f1_txt = f"{met['f1']:.4f}" if met.get("f1") is not None else "  -  "
        kept = f"{met['gold_kept_rate']:.3f}" if met.get("gold_kept_rate") is not None else "  -  "
        print(f"{run_id:18s}{met['recall@1']:7.4f}{met['recall@5']:7.4f}{met['mrr']:7.4f}"
              f"{met['ndcg@5']:7.4f}{f1_txt:>7s}{kept:>7s}{rerank_ms:9.1f}"
              f"{met['latency_ms']:8.1f}  → {path.name}  ({time.time() - t0:.1f}s)")
        log(RE.format_line(f"[{run_id}]", met, met["recall@5_ci"]) +
            f" F1={f1_txt} rerank={rerank_ms:.1f}ms total={met['latency_ms']:.1f}ms")
        summary.append({"run_id": run_id, "depth": depth, **met})
        reranker.last_stats = {}

    # ---------------- 汇总 ----------------
    dump_json({"experiment": "E4", "split": args.split, "rows": summary,
               "config": {"alpha": args.alpha, "pool": args.pool, "topn": args.topn,
                          "depths": depths, "reader": not args.no_reader,
                          "max_chars": int(args.max_chars)}},
              path_of(cfg, "results_raw") / "E4_summary.json")

    table = [{
        "run_id": r["run_id"], "depth": r["depth"], "rerank": r["depth"] is not None,
        "recall@1": r.get("recall@1"), "recall@5": r.get("recall@5"),
        "recall@5_lo": (r.get("recall@5_ci") or [None, None])[0],
        "recall@5_hi": (r.get("recall@5_ci") or [None, None])[1],
        "mrr": r.get("mrr"), "ndcg@5": r.get("ndcg@5"),
        "f1": r.get("f1"), "em": r.get("em"),
        "f1_lo": (r.get("f1_ci") or [None, None])[0] if r.get("f1_ci") else None,
        "f1_hi": (r.get("f1_ci") or [None, None])[1] if r.get("f1_ci") else None,
        "gold_kept_rate": r.get("gold_kept_rate"),
        "rerank_ms": r.get("rerank_ms"),
        "reader_ms": r.get("reader_ms"),
        "retrieval_ms": r.get("retrieval_ms"),
        "total_ms": r.get("latency_ms"),
    } for r in summary]
    tab = write_table("E4_rerank", table, cfg)
    print("-" * 96)
    print(f"[OK] {len(summary)} 个 arm 全部落盘 → results/raw/E4_*.json ；表 → {tab.name}")
    log(f"[done] {len(summary)} arms，总耗时 {time.time() - t_all:.1f}s")
    log.close()

    if not args.no_plot:
        try:
            from eval import plots_c
            fig = plots_c.plot_e4(cfg)
            if fig:
                print(f"[fig] {fig.relative_to(ROOT)}")
        except Exception as exc:  # 画图失败不该让实验白跑
            print(f"[warn] 画图失败：{type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
