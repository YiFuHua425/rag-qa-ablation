#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E7：CPU 与 GPU 的延迟对照 + 指标不变性验证。

为什么单独一个脚本而不是改 run_e4：
    run_e4 的延迟口径被硬编码为 ``latency.device=cpu``（CPU 单线程），
    这是报告 E4 的正式口径。直接改它会让已有结论不可复现。
    本脚本不动任何既有输出，只产出 ``results/tab/E7_gpu_cpu.csv``。

它回答两个问题：
    1. 同一配置换到 GPU，**指标是否变**？（理论不变，这里给实测）
    2. 换到 GPU，**延迟降多少**？（按 run_e4 同一口径：batch=1、单线程探针、取中位数）

用法：
    python eval/run_e7_gpu.py --split smoke --limit-queries 20
    python eval/run_e7_gpu.py --devices cpu,cuda --depths 20,50,100
    python eval/run_e7_gpu.py --devices cuda          # 只测 GPU
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description="E7：CPU vs GPU 延迟对照")
    ap.add_argument("--split", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--limit-queries", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.2, help="沿用 E2 最优融合权重")
    ap.add_argument("--pool", type=int, default=100, help="候选池深度（≥ 最大 depth）")
    ap.add_argument("--depths", default="20,50,100")
    ap.add_argument("--topn", type=int, default=5)
    ap.add_argument("--devices", default="cpu,cuda")
    ap.add_argument("--latency-queries", type=int, default=2, help="延迟探针用几题")
    ap.add_argument("--latency-repeat", type=int, default=2)
    ap.add_argument("--no-reader", action="store_true")
    ap.add_argument("--no-cache", action="store_true",
                    help="强制重算全池打分（做设备对照必须开：否则第二个设备会命中第一个的缓存，"
                         "「指标一致」就成了循环论证）")
    ap.add_argument("--out", default="results/tab/E7_gpu_cpu.csv")
    args = ap.parse_args()

    import torch
    from eval import c_pipeline as CP
    from eval.io_utils import load_config
    from reader.extractive_qa import ExtractiveReader
    from rerank.cross_encoder import Reranker

    cfg = load_config()
    depths = [int(x) for x in str(args.depths).split(",") if x.strip()]
    if max(depths) > args.pool:
        raise SystemExit(f"--pool({args.pool}) 必须 ≥ 最大 depth({max(depths)})")

    score_threads = max(1, (torch.get_num_threads() or 1))   # 算分线程（与 run_e4 同思路）
    rerank_cfg = cfg.get("rerank", {}) or {}
    reader_cfg = cfg.get("reader", {}) or {}
    ctx_cfg = cfg.get("context", {}) or {}

    print("=" * 92)
    print(" E7 · CPU vs GPU 延迟对照（指标不变性验证）")
    print("=" * 92)
    print(f" torch {torch.__version__}　CUDA {torch.version.cuda or '不可用'}"
          f"　cuda.is_available()={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f" 设备：{torch.cuda.get_device_name(0)}")
    print(f" split={args.split}　题数={args.limit_queries or '全部'}　α={args.alpha}"
          f"　pool={args.pool}　depths={depths}")
    print(f" 延迟口径：batch=1、单线程探针、{args.latency_repeat} 轮取中位数；"
          f"算分用 {score_threads} 线程")

    t_all = time.perf_counter()
    bundle = CP.build_candidates(cfg, split=args.split, alpha=args.alpha,
                                 depth=args.pool, limit=args.limit_queries)
    print(f" 候选就绪：{len(bundle)} 题　检索延迟 {CP.retrieval_latency(bundle):.2f} ms/query\n")

    rows: list[dict] = []
    ranked_by_dev: dict[str, dict[int, list[list[str]]]] = {}
    reader_stats: dict[str, dict] = {}

    for dev in [d.strip() for d in args.devices.split(",") if d.strip()]:
        if dev.startswith("cuda") and not torch.cuda.is_available():
            print(f"[skip] 设备 {dev} 不可用（torch 无 CUDA 支持）")
            continue
        print(f"--- 设备 {dev} ---")
        reranker = Reranker(
            model_name=rerank_cfg.get("model", "BAAI/bge-reranker-base"),
            depth=max(depths), topn=args.topn, device=dev,
            max_length=int(rerank_cfg.get("max_length", 512)),
            threads=1, score_threads=score_threads,
        ).set_corpus(bundle.docs)

        t0 = time.perf_counter()
        pooled = CP.rerank_pool(bundle, reranker, pool=args.pool,
                                use_cache=not args.no_cache)
        pool_sec = time.perf_counter() - t0
        print(f"  全池打分（{len(bundle)}×{args.pool}）{pool_sec:.1f} s"
              f"{'　[强制重算]' if args.no_cache else ''}")

        ranked_by_dev[dev] = {}
        probe_idx = range(min(args.latency_queries, len(bundle)))
        for depth in depths:
            ranked = CP.topk_from_pool(pooled, depth, args.topn)
            ranked_by_dev[dev][depth] = ranked
            met = CP.rank_metrics(ranked, bundle.golds, ndcg_k=args.topn)
            reranker.depth = depth
            lat = [reranker.benchmark_latency(bundle.questions[i], bundle.candidates[i],
                                              repeat=args.latency_repeat)
                   for i in probe_idx]
            ms = CP.median(lat)
            rows.append({"device": dev, "depth": depth,
                         "recall@1": met["recall@1"], "recall@5": met["recall@5"],
                         "mrr": met["mrr"], "ndcg@5": met["ndcg@5"],
                         "rerank_ms": round(ms, 2),
                         "per_cand_ms": round(ms / depth, 3),
                         "pool_sec": round(pool_sec, 2)})
            print(f"  depth={depth:<4} R@1 {met['recall@1']:.4f}  R@5 {met['recall@5']:.4f}  "
                  f"MRR {met['mrr']:.4f}  重排 {ms:.1f} ms（{ms/depth:.2f} ms/候选）")

        if not args.no_reader:
            reader = ExtractiveReader(
                model_name=reader_cfg.get("model", "uer/roberta-base-chinese-extractive-qa"),
                max_length=int(reader_cfg.get("max_length", 512)), device=dev,
                batch_size=int(reader_cfg.get("batch_size", 16)),
                threads=1, score_threads=score_threads,
            )
            probe = [(bundle.questions[i], bundle.doc_map[bundle.golds[i]])
                     for i in probe_idx]
            r_ms = reader.benchmark_latency(probe, repeat=args.latency_repeat)
            t0 = time.perf_counter()
            reading = CP.read_answers(reader, bundle, ranked_by_dev[dev][max(depths)],
                                      order=str(ctx_cfg.get("order", "natural")),
                                      max_chars=int(ctx_cfg.get("max_chars", 2000)),
                                      progress=f"reader-{dev}")
            read_sec = time.perf_counter() - t0
            sm = CP.summarize_reading(reading)
            reader_stats[dev] = {"reader_ms": round(r_ms, 1),
                                 "f1": sm.get("f1"), "em": sm.get("em"),
                                 "read_sec": round(read_sec, 1)}
            print(f"  阅读器：{r_ms:.1f} ms/query　F1 {sm.get('f1'):.4f}　"
                  f"EM {sm.get('em'):.4f}　{len(bundle)} 题共 {read_sec:.1f} s")
        print()

    # ------------------------------------------------ 指标一致性
    devs = list(ranked_by_dev)
    if len(devs) >= 2:
        a, b = devs[0], devs[1]
        print("=" * 92)
        print(f" 指标一致性：{a} vs {b}")
        print("=" * 92)
        for depth in depths:
            same = sum(1 for x, y in zip(ranked_by_dev[a][depth], ranked_by_dev[b][depth]) if x == y)
            n = len(ranked_by_dev[a][depth])
            print(f"  depth={depth:<4} top{args.topn} 名单完全一致的题：{same}/{n}"
                  f"（{same/n:.1%}）")
        if reader_stats and len(reader_stats) >= 2:
            fa, fb = reader_stats[a].get("f1"), reader_stats[b].get("f1")
            if fa is not None and fb is not None:
                print(f"  阅读器 F1：{a} {fa:.4f}　{b} {fb:.4f}　差 {abs(fa-fb):.6f}")
        print()

        # ------------------------------------------------ 延迟加速比
        print(f" 延迟加速比（{a} → {b}）")
        by = {(r["device"], r["depth"]): r for r in rows}
        for depth in depths:
            ra, rb = by.get((a, depth)), by.get((b, depth))
            if ra and rb and rb["rerank_ms"]:
                print(f"  depth={depth:<4} {ra['rerank_ms']:>9.1f} ms → {rb['rerank_ms']:>8.1f} ms"
                      f"　×{ra['rerank_ms']/rb['rerank_ms']:.1f}")
        if reader_stats and len(reader_stats) >= 2:
            la, lb = reader_stats[a]["reader_ms"], reader_stats[b]["reader_ms"]
            if lb:
                print(f"  阅读器    {la:>9.1f} ms → {lb:>8.1f} ms　×{la/lb:.1f}")
        print()

    # ------------------------------------------------ 落盘
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["device", "depth", "recall@1", "recall@5", "mrr", "ndcg@5",
              "rerank_ms", "per_cand_ms", "pool_sec"]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    print(f"[out] {out.relative_to(ROOT)}（{len(rows)} 行）")
    print(f"全程 {time.perf_counter() - t_all:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())