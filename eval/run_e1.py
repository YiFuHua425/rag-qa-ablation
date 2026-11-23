#!/usr/bin/env python
"""E1 基线实验入口 —— 负责人：A（集成）

用法::

    python eval/run_e1.py --dry-run          # 零依赖，验证骨架（第 0 棒验收）
    python eval/run_e1.py --split smoke      # 真实运行，需 B/C/D 模块就位

--dry-run 的设计：用**真实指标代码** + **假检索结果**，所以它同时验证
  (1) A 的 pipeline 接线是否正确
  (2) D 的 metrics 是否能跑
不做的事：不下载模型、不读数据、不碰 torch —— clone 下来就能跑。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 管道/重定向时 Windows 默认用 GBK 编码，中文会乱码；统一按 UTF-8 输出
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from eval.metrics import (  # noqa: E402
    bootstrap_ci,
    em,
    f1,
    mean,
    mrr,
    ndcg_at_k,
    recall_at_k,
)

RAW_DIR = ROOT / "results" / "raw"


# --------------------------------------------------------------- dry-run 世界

def _dry_world(n_questions: int = 30, n_docs: int = 60, seed: int = 42):
    """造一个可控的假世界：金标排名的分布模仿真实稠密检索的行为。"""
    rng = random.Random(seed)
    doc_ids = [f"p_{i:05d}" for i in range(n_docs)]
    weights = {1: 0.62, 2: 0.16, 3: 0.08, 4: 0.05, 5: 0.05, 6: 0.02, 8: 0.01, 10: 0.01}
    ranks = list(weights)
    rank_probs = [weights[r] for r in ranks]

    questions = []
    for i in range(n_questions):
        gold = rng.choice(doc_ids)
        rank = rng.choices(ranks, weights=rank_probs, k=1)[0]
        distractors = [d for d in doc_ids if d != gold]
        rng.shuffle(distractors)
        ranked = distractors[: rank - 1] + [gold] + distractors[rank - 1 :]
        hit = rng.random() < 0.78           # 阅读器答对的概率
        questions.append({
            "qid": f"dry_{i:04d}",
            "question": f"示例问题 {i}",
            "gold_doc_id": gold,
            "answer_text": f"示例答案{i}",
            "ranked": ranked[:10],
            "pred": f"示例答案{i}" if hit else "",
        })
    return doc_ids, questions


def run_dry(n_questions: int = 30, topk: int = 5, seed: int = 42,
            run_id: str = "E1_dryrun", label: str = "稠密检索基线") -> dict:
    t0 = time.perf_counter()
    doc_ids, rows = _dry_world(n_questions=n_questions, seed=seed)
    elapsed = (time.perf_counter() - t0) * 1000.0

    r1, r5, rr, nd, ems, f1s = [], [], [], [], [], []
    per_question = []
    for row in rows:
        ranked, gold = row["ranked"], row["gold_doc_id"]
        r1.append(recall_at_k(ranked, gold, 1))
        r5.append(recall_at_k(ranked, gold, topk))
        rr.append(mrr(ranked, gold))
        nd.append(ndcg_at_k(ranked, gold, topk))
        ems.append(em(row["pred"], row["answer_text"]))
        f1s.append(f1(row["pred"], row["answer_text"]))
        per_question.append({
            "qid": row["qid"], "ranked": ranked, "gold": gold,
            "hit": gold in ranked[:topk], "pred": row["pred"],
            "f1": f1(row["pred"], row["answer_text"]),
        })

    lo, hi = bootstrap_ci(r5)
    result = {
        "run_id": run_id,
        "config": {"retriever": "dense", "topk": topk, "rerank": False, "dry_run": True},
        "meta": {"split": "dry", "n_questions": len(rows), "seed": seed,
                 "n_docs": len(doc_ids), "python": sys.version.split()[0]},
        "metrics": {
            "recall@1": mean(r1),
            f"recall@{topk}": mean(r5),
            "mrr": mean(rr),
            f"ndcg@{topk}": mean(nd),
            "em": mean(ems),
            "f1": mean(f1s),
            "latency_ms": round(elapsed / max(1, len(rows)), 3),
            f"recall@{topk}_ci": [lo, hi],
        },
        "per_question": per_question,
    }
    return result


# ------------------------------------------------------------------ 真实运行

def run_real(split: str, topk: int, seed: int = 42, backend: str = "auto",
             model_path: str | None = None, limit: int = 0, use_reader: bool = True) -> dict:
    """第 2 棒：把 B 的检索接进 A 的入口，跑出 E1 基线。

    C 的 ``ExtractiveReader`` 还是 STUB 时自动降级：Recall/MRR/nDCG 照常出数，
    F1/EM 记 null 并在 meta 里写明原因——**检索那半边不阻塞任何人**。
    """
    import statistics
    from eval import io_utils as IU
    from eval.retrieval_eval import metrics_dict, per_question_records, recall_ci
    from retrieval import BgeEncoder, FaissIndex

    cfg = IU.load_config()
    docs, queries = IU.load_split(split, cfg)
    if limit:
        queries = queries[: int(limit)]
    dense = cfg.get("retrieval", {}).get("dense", {})
    print(f"[E1] split={split}  docs={len(docs)}  queries={len(queries)}  开始编码/建索引 …")

    enc = BgeEncoder(
        model_name=dense.get("model", "BAAI/bge-small-zh-v1.5"),
        query_prefix=dense.get("query_prefix", ""),
        batch_size=int(dense.get("encode_batch", 64)),
        model_path=model_path,
        backend=backend,
    )
    # 第 2 棒原文要求「BGE 编码全库 + FAISS 建索引」：faiss 装了就用它（IndexFlatIP），
    # 没装则自动退 numpy 内积。两者 top-100 集合 100% 一致、指标逐位一致（见 results/tab/E2_backend.csv）。
    index = FaissIndex(use_faiss=True)
    index.build_from_encoder(enc, docs, tag="docs")
    print(f"[E1] 索引就绪：{len(index.doc_ids)} 篇 × {index.dim} 维（backend={index.backend}）")

    fusion = cfg.get("retrieval", {}).get("fusion", {})
    depth = max(int(topk), int(fusion.get("dense_depth", 100)))
    q_texts = [q["question"] for q in queries]
    q_emb = enc.encode_queries(q_texts, show_progress=True)
    rankings = index.search_ids_batch(q_emb, depth)
    golds = [q["gold_doc_id"] for q in queries]

    metrics = metrics_dict(rankings, golds, ks=(1, 5, 10, 20), ndcg_k=topk)
    ci = recall_ci(rankings, golds, k=topk,
                   n_boot=int(cfg.get("eval", {}).get("bootstrap_n", 1000)),
                   alpha=float(cfg.get("eval", {}).get("alpha", 0.05)))

    # 延迟：30 题 × 3 轮取中位数，含查询编码 + 检索
    probe = min(30, len(queries))
    times: list[float] = []
    for _ in range(3):
        for i in range(probe):
            t0 = time.perf_counter()
            v = enc.encode_queries([q_texts[i]])
            index.search(v[0], topk)
            times.append((time.perf_counter() - t0) * 1000.0)
    latency = statistics.median(times)

    # ---- 阅读器（C 的模块）：STUB 时降级，不让 F1 卡住检索数字
    preds: list[str] = []
    reader_state = "stub"
    try:
        from reader import extractive_qa as _rq

        if use_reader and not getattr(_rq, "STUB", True):
            from reader.context_builder import build_context

            doc_map = {d["doc_id"]: d for d in docs}
            # 阅读器默认单线程；这里开多线程算分（只影响速度，不改指标）。
            # 上下文 2000 字要切多窗，单线程实测约 3.9 s/题，500 题要 32 分钟。
            import os as _os

            _st = max(1, (_os.cpu_count() or 3) - 2)
            reader = _rq.ExtractiveReader(threads=1, score_threads=_st)
            ctx_cfg = cfg.get("context", {})
            import time as _t

            n_q = len(queries)
            print(f"  [reader] 开始逐题推理：{n_q} 题（{_st} 线程算分）", flush=True)
            _t0 = _t.perf_counter()
            for i, (q, ranked) in enumerate(zip(queries, rankings), 1):
                ctx, _used = build_context(
                    ranked[:topk], doc_map,
                    order=ctx_cfg.get("order", "natural"),
                    max_chars=int(ctx_cfg.get("max_chars", 2000)),
                    gold_doc_id=None,
                )
                pred, _score = reader.answer(q["question"], ctx)
                preds.append(pred)
                # 500 题闷头跑几十分钟、期间零输出，会被误判成"卡死"，所以每 25 题报一次
                if i % 25 == 0 or i == n_q:
                    el = _t.perf_counter() - _t0
                    eta = el / i * (n_q - i)
                    print(f"  [reader] {i}/{n_q}　已用 {el:.0f}s　预计还需 {eta:.0f}s", flush=True)
            reader_state = "ok"
    except (ImportError, NotImplementedError, TypeError, AttributeError,
            OSError, RuntimeError) as exc:      # C: 权重缺失时也降级，而不是让整条命令崩
        reader_state = f"skipped: {type(exc).__name__}"

    if reader_state == "ok" and len(preds) == len(queries):
        metrics["em"] = mean([em(p, q["answer_text"]) for p, q in zip(preds, queries)])
        metrics["f1"] = mean([f1(p, q["answer_text"]) for p, q in zip(preds, queries)])
    else:
        metrics["em"] = None
        metrics["f1"] = None

    metrics["latency_ms"] = round(latency, 3)
    metrics[f"recall@{topk}_ci"] = ci

    rows = per_question_records([q["qid"] for q in queries], rankings, golds, top_show=depth)
    for i, row in enumerate(rows):
        row["pred"] = preds[i] if i < len(preds) else ""
        row["f1"] = None if not preds else f1(preds[i], queries[i]["answer_text"])

    run_id = f"E1_dense_top{topk}"
    out = IU.write_run(
        run_id,
        config={"retriever": "dense", "topk": topk, "depth": depth,
                "rerank": False, "reader": reader_state,
                "model": enc.model_name,
                "model_file": Path(str(enc.model_file)).name,
                "index_backend": index.backend},
        metrics=metrics,
        per_question=rows,
        meta={"split": split, "n_questions": len(queries), "n_docs": len(docs),
              "seed": seed, "reader": reader_state,
              "note": "int8 权重" if model_path and "int8" in str(model_path) else "auto 选中的权重"},
        cfg=cfg,
    )

    f1_txt = f"{metrics['f1']:.3f}" if metrics["f1"] is not None else "待 C（reader STUB）"
    print("-" * 62)
    print(f"Recall@1        {metrics['recall@1']:.3f}")
    print(f"Recall@{topk}        {metrics[f'recall@{topk}']:.3f}   95%CI [{ci[0]:.3f}, {ci[1]:.3f}]")
    print(f"MRR             {metrics['mrr']:.3f}")
    print(f"nDCG@{topk}         {metrics[f'ndcg@{topk}']:.3f}")
    print(f"F1              {f1_txt}")
    print(f"latency         {latency:.2f} ms/query  （含查询编码 + 检索，30 题×3 轮中位数）")
    print("-" * 62)
    print(f"→ {out.relative_to(ROOT)}")
    if reader_state != "ok":
        print(f"[!] 阅读器未参与（{reader_state}）：F1/EM 记 null。")
        print("    检查 reader/extractive_qa.py 是否 READY，以及权重是否就位："
              "python scripts/download_models_c.py --only reader")
    return {"run_id": run_id, "metrics": metrics, "path": str(out)}


# ---------------------------------------------------------------------- CLI

def main() -> int:
    ap = argparse.ArgumentParser(description="E1 稠密检索基线")
    ap.add_argument("--dry-run", action="store_true", help="零依赖验证骨架")
    ap.add_argument("--split", default="smoke", choices=["smoke", "full", "dry"])
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=30, help="dry-run 的题目数")
    ap.add_argument("--encoder", default="auto", choices=["auto", "onnx", "st", "hash"],
                    help="稠密编码后端；auto = onnx → st → hash")
    ap.add_argument("--model-path", default=None,
                    help="钉住权重文件（复现报告数字请给 models/bge-small-zh-v1.5-onnx/model_int8.onnx）")
    ap.add_argument("--limit-queries", type=int, default=0, help="只跑前 N 题（调试用）")
    ap.add_argument("--no-reader", action="store_true", help="只跑检索，不调用阅读器")
    args = ap.parse_args()

    if not args.dry_run and args.split != "dry":
        run_real(args.split, args.topk, seed=args.seed, backend=args.encoder,
                 model_path=args.model_path, limit=args.limit_queries,
                 use_reader=not args.no_reader)
        return 0

    res = run_dry(n_questions=args.n, topk=args.topk, seed=args.seed)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / f"{res['run_id']}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    m = res["metrics"]
    k = args.topk
    print(f"[E1] {res['config']['retriever']}  split=dry  "
          f"n={res['meta']['n_questions']}  seed={res['meta']['seed']}")
    print("-" * 62)
    print(f"Recall@1        {m['recall@1']:.3f}")
    print(f"Recall@{k}        {m[f'recall@{k}']:.3f}")
    print(f"MRR             {m['mrr']:.3f}")
    print(f"nDCG@{k}         {m[f'ndcg@{k}']:.3f}")
    print(f"EM              {m['em']:.3f}")
    print(f"F1              {m['f1']:.3f}")
    lo, hi = m[f"recall@{k}_ci"]
    print(f"Recall@{k} 95%CI  [{lo:.3f}, {hi:.3f}]")
    print(f"latency         {m['latency_ms']:.3f} ms/query  (dry-run 假数据，无意义)")
    print("-" * 62)
    print(f"→ {out.relative_to(ROOT)}")
    print("[OK] 骨架通过：接口可用，可以开始接自己的模块。")
    print("     下一步：B 换 bm25.py，C 换 reader/，D 换 data/（见 INTERFACE.md）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())