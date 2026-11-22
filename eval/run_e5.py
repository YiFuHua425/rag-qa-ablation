#!/usr/bin/env python
"""E5：上下文位置偏差 —— 负责人：C（接力棒第 5 棒，全案最值钱的实验）

    python eval/run_e5.py --split smoke --limit-queries 300
    python eval/run_e5.py --split smoke --limit-queries 50 --no-plot   # 冒烟

**只改 ``context_builder.build_context(order=...)``，检索与重排一律不动**：
同一批题、同一份候选、同一个阅读器，唯一的变量是金标段落在上下文里的位置。

================  ==================================================
order             含义
================  ==================================================
``natural``       检索/重排给出的顺序（对照）
``gold_first``    金标挪到第 1 段
``gold_middle``   金标挪到正中间
``gold_last``     金标挪到最后一段
================  ==================================================

再叠一档 ``max_chars``（2000 / 4000）看"上下文越长是否越糟"。**截断会吃掉尾部段落**，
所以每个 run 都同时记 ``gold_kept_rate``（金标是否真的进了上下文）与 ``gold_kept_f1``
（只在金标进得去的题上算 F1）——这两个数分开看，才能把
「位置效应」和「截断把答案裁掉了」区分开，这也是 INTERFACE 要求
``build_context`` 必须返回 ``used_doc_ids`` 的原因。

产出::

    results/raw/E5_pos_<order>_c<max_chars>.json
    results/raw/E5_summary.json
    results/tab/E5_position.csv
    results/fig/E5_position.png
    results/log/E5_run.log
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
from eval import metrics as M  # noqa: E402
from eval import retrieval_eval as RE  # noqa: E402
from eval.io_utils import Logger, dump_json, load_config, path_of, write_run, write_table  # noqa: E402


def _subset(rows, flag: str = "gold_in_candidates"):
    return [r for r in rows if r.get(flag)]


def main() -> int:
    ap = argparse.ArgumentParser(description="E5: 上下文位置偏差")
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--limit-queries", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--pool", type=int, default=100)
    ap.add_argument("--depth", type=int, default=50, help="重排深度（固定，不当变量）")
    ap.add_argument("--topn", type=int, default=5, help="进入上下文的段落数上限")
    ap.add_argument("--orders", default="natural,gold_first,gold_middle,gold_last")
    ap.add_argument("--max-chars", default="2000,4000")
    ap.add_argument("--rerank", default="on", choices=["on", "off"])
    ap.add_argument("--reader-model", default=None)
    ap.add_argument("--rerank-model", default=None)
    ap.add_argument("--rerank-backend", default="auto", choices=["auto", "hf", "onnx", "lexical"],
                    help="lexical 只用于无模型的环境验证链路，数字不可进报告")
    ap.add_argument("--encoder", default="auto", choices=["auto", "onnx", "st", "hash"])
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--no-reader", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    orders = [o.strip() for o in str(args.orders).split(",") if o.strip()]
    char_grid = [int(x) for x in str(args.max_chars).split(",") if x.strip()]
    seed = int((cfg.get("project", {}) or {}).get("seed", 42))
    boot_n = int((cfg.get("eval", {}) or {}).get("bootstrap_n", 1000))
    boot_alpha = float((cfg.get("eval", {}) or {}).get("alpha", 0.05))
    lat_cfg = cfg.get("latency", {}) or {}
    threads = int(lat_cfg.get("threads", 1))
    rerank_cfg = cfg.get("rerank", {}) or {}
    reader_cfg = cfg.get("reader", {}) or {}
    use_rerank = args.rerank == "on"
    score_threads = max(1, (os.cpu_count() or 3) - 2)

    log = Logger("E5_run", cfg)
    log(f"[E5] split={args.split} alpha={args.alpha} depth={args.depth} rerank={use_rerank} "
        f"orders={orders} max_chars={char_grid} topn={args.topn}")

    t_all = time.time()
    bundle = CP.build_candidates(cfg, split=args.split, alpha=args.alpha, depth=args.pool,
                                 encoder_backend=args.encoder, model_path=args.model_path,
                                 limit=args.limit_queries, use_cache=not args.no_cache)
    log(f"[data] n_docs={len(bundle.docs)} n_questions={len(bundle)} | {bundle.info}")

    # ---------------- 固定检索/重排，只留位置一个变量 ----------------
    ranked = [[d for d, _ in cand[: args.topn]] for cand in bundle.candidates]
    reranker = None
    if use_rerank:
        from rerank.cross_encoder import Reranker

        reranker = Reranker(
            model_name=rerank_cfg.get("model", "BAAI/bge-reranker-base"),
            depth=args.depth, topn=args.topn, device=str(lat_cfg.get("device", "cpu")),
            max_length=int(rerank_cfg.get("max_length", 512)),
            model_path=args.rerank_model, threads=threads, backend=args.rerank_backend,
            score_threads=score_threads,
        ).set_corpus(bundle.docs)
        # 与 E4 共用同一份打分缓存（同 α / 同 pool / 同题目时直接命中，省一遍全量前向）
        pooled = CP.rerank_pool(bundle, reranker, pool=max(args.pool, args.depth),
                                use_cache=not args.no_cache)
        ranked = CP.topk_from_pool(pooled, args.depth, args.topn)
        log(f"[rerank] {reranker} | depth={args.depth} pool={args.pool}")

    met_retr = CP.rank_metrics(ranked, bundle.golds, ndcg_k=5)
    log(RE.format_line("[fixed-retrieval]", met_retr))

    reader = None
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

    # ---------------- 位置 × 上下文长度 ----------------
    summary = []
    print("-" * 104)
    print(f"{'run_id':26s}{'F1':>8s}{'F1(kept)':>9s}{'kept':>7s}{'pos':>6s}"
          f"{'chars':>7s}{'F1_CI':>18s}  状态")
    print("-" * 104)

    for max_chars in char_grid:
        for order in orders:
            run_id = f"E5_pos_{order}_c{max_chars}"
            t0 = time.time()
            if reader is not None:
                reading = CP.read_answers(reader, bundle, ranked, order=order,
                                          max_chars=max_chars,
                                          progress=f"{run_id}")
                rows = reading["rows"]
                met = {"f1": round(reading["f1"], 6), "em": round(reading["em"], 6)}
                met["f1_ci"] = CP.bootstrap_f1(rows, boot_n, boot_alpha, seed)
                kept = [r for r in rows if r["gold_used"]]
                met["gold_kept_rate"] = round(len(kept) / max(1, len(rows)), 6)
                met["gold_kept_f1"] = round(M.mean([r["f1"] for r in kept]), 6) if kept else 0.0
                retrieved = [r for r in rows if r["gold_in_candidates"]]
                met["n_retrieved"] = len(retrieved)
                met["f1_retrieved"] = round(M.mean([r["f1"] for r in retrieved]), 6) if retrieved else 0.0
                met["gold_position_mean"] = round(M.mean([r["gold_position"] for r in kept]), 3) if kept else 0.0
                met["ctx_chars_mean"] = round(M.mean([r["ctx_chars"] for r in rows]), 1)
            else:
                # 不跑阅读器时也算截断统计：只需要 build_context 的诊断信息
                from reader.context_builder import build_context, get_last_stats

                kept = 0
                chars = []
                for i, row in enumerate(ranked):
                    ctx, _used = build_context(row, bundle.doc_map, order=order,
                                               max_chars=max_chars,
                                               gold_doc_id=bundle.golds[i])
                    st = get_last_stats()
                    kept += int(bool(st["gold_used"]))
                    chars.append(st["chars"])
                rows = [{"qid": q, "f1": None, "gold_used": None}
                        for q in bundle.qids]
                met = {"f1": None, "em": None, "f1_ci": None,
                       "gold_kept_rate": round(kept / max(1, len(ranked)), 6),
                       "gold_kept_f1": None, "n_retrieved": None, "f1_retrieved": None,
                       "gold_position_mean": None,
                       "ctx_chars_mean": round(M.mean(chars), 1)}

            met.update({k: met_retr[k] for k in met_retr})
            met["rerank"] = use_rerank
            per_question = []
            for i, rd in enumerate(rows):
                per_question.append({
                    "qid": rd["qid"], "ranked": rd.get("ranked", ranked[i]),
                    "gold": bundle.golds[i],
                    "hit": bundle.golds[i] in ranked[i],
                    "pred": rd.get("pred", ""), "f1": rd.get("f1"),
                    "gold_used": rd.get("gold_used"),
                    "gold_position": rd.get("gold_position"),
                    "ctx_chars": rd.get("ctx_chars"),
                })
            path = write_run(
                run_id,
                config={"retriever": "rrf", "alpha": args.alpha, "rerank": use_rerank,
                        "depth": args.depth if use_rerank else None, "topn": args.topn,
                        "order": order, "max_chars": max_chars,
                        "reader": "off" if reader is None else str(reader_cfg.get("model"))},
                metrics=met, per_question=per_question,
                meta={"split": args.split, "n_questions": len(bundle),
                      "latency": {"threads": threads,
                                  "device": lat_cfg.get("device", "cpu")}},
                cfg=cfg)
            f1 = f"{met['f1']:.4f}" if met["f1"] is not None else "   -   "
            f1k = f"{met['gold_kept_f1']:.4f}" if met["gold_kept_f1"] is not None else "   -   "
            ci = met.get("f1_ci") or [None, None]
            ci_txt = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci[0] is not None else "        -        "
            print(f"{run_id:26s}{f1:>8s}{f1k:>9s}{met['gold_kept_rate']:7.3f}"
                  f"{(met['gold_position_mean'] or 0):6.2f}{met['ctx_chars_mean']:7.0f}"
                  f"{ci_txt:>18s}  → {path.name} ({time.time() - t0:.0f}s)")
            log(RE.format_line(f"[{run_id}]", met, met.get("f1_ci")) +
                f" kept={met['gold_kept_rate']:.3f}")
            summary.append({"run_id": run_id, "order": order, "max_chars": max_chars, **met})

    dump_json({"experiment": "E5", "split": args.split, "rows": summary,
               "config": {"alpha": args.alpha, "depth": args.depth, "rerank": use_rerank,
                          "topn": args.topn, "orders": orders, "max_chars": char_grid,
                          "n_questions": len(bundle), "retrieval": met_retr}},
              path_of(cfg, "results_raw") / "E5_summary.json")

    table = [{
        "run_id": r["run_id"], "order": r["order"], "max_chars": r["max_chars"],
        "n_questions": len(bundle), "n_retrieved": r.get("n_retrieved"),
        "f1": r.get("f1"), "f1_lo": (r.get("f1_ci") or [None, None])[0] if r.get("f1_ci") else None,
        "f1_hi": (r.get("f1_ci") or [None, None])[1] if r.get("f1_ci") else None,
        "em": r.get("em"), "f1_retrieved": r.get("f1_retrieved"),
        "gold_kept_rate": r.get("gold_kept_rate"), "gold_kept_f1": r.get("gold_kept_f1"),
        "gold_position_mean": r.get("gold_position_mean"),
        "ctx_chars_mean": r.get("ctx_chars_mean"),
    } for r in summary]
    tab = write_table("E5_position", table, cfg)
    print("-" * 104)
    print(f"[OK] {len(summary)} 个 run 落盘 → results/raw/E5_*.json ；表 → {tab.name}")
    log(f"[done] {len(summary)} runs，总耗时 {time.time() - t_all:.1f}s")
    log.close()

    if not args.no_plot:
        try:
            from eval import plots_c
            fig = plots_c.plot_e5(cfg)
            if fig:
                print(f"[fig] {fig.relative_to(ROOT)}")
        except Exception as exc:
            print(f"[warn] 画图失败：{type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
