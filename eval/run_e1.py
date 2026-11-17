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

def run_real(split: str, topk: int) -> dict:
    """TODO(A)：第 2 棒打通。
    docs  = load_jsonl(data/corpus.jsonl)
    dev   = load_jsonl(data/dev.jsonl)  (smoke 用 data/dev_smoke.jsonl)
    enc   = BgeEncoder(...); idx = FaissIndex(...)
    for q in dev:
        ranked = idx.search(enc.encode_query(q["question"]), topk*20)
        ctx, used = build_context(...)
        pred, score = reader.answer(q["question"], ctx)
    """
    corpus = ROOT / "data" / "corpus.jsonl"
    dev = ROOT / "data" / ("dev_smoke.jsonl" if split == "smoke" else "dev.jsonl")
    missing = [str(p) for p in (corpus, dev) if not p.exists()]
    if missing:
        print("[X] 缺少数据文件：" + ", ".join(missing), file=sys.stderr)
        print("    → 这是第 1 棒（D）的交付物；先跑 "
              "`python eval/check_data.py` 看具体要求。", file=sys.stderr)
        print("    → 或者先用 `--dry-run` 验证骨架。", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(
        "[!] 数据齐了，但真实 pipeline 尚未接线（第 2 棒的活）。\n"
        "    A：在 eval/run_e1.py::run_real() 里接 B 的 BgeEncoder/FaissIndex、\n"
        "       C 的 build_context/ExtractiveReader、D 的 metrics。"
    )


# ---------------------------------------------------------------------- CLI

def main() -> int:
    ap = argparse.ArgumentParser(description="E1 稠密检索基线")
    ap.add_argument("--dry-run", action="store_true", help="零依赖验证骨架")
    ap.add_argument("--split", default="smoke", choices=["smoke", "full", "dry"])
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=30, help="dry-run 的题目数")
    args = ap.parse_args()

    if not args.dry_run and args.split != "dry":
        run_real(args.split, args.topk)   # 内部会 SystemExit
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