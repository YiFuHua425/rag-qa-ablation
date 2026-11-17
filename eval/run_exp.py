#!/usr/bin/env python
"""批量实验入口 —— 负责人：A（集成）

用法::

    python eval/run_exp.py --list                 # 看全部实验与配置矩阵
    python eval/run_exp.py --exp smoke --dry-run  # 零依赖：跑通"配置可切换"的骨架
    python eval/run_exp.py --exp E2 --split smoke # 真实运行（第 3 棒后可用）

设计要点：**一个入口、多套配置**。第 3 棒的验收标准不是"功能写完"，
而是"仅靠命令行参数就能变出 6 种配置并各自落盘"——没有这个，
第 4 棒的矩阵对照跑不起来。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from eval.run_e1 import run_dry  # noqa: E402

RAW_DIR = ROOT / "results" / "raw"

# 实验 → 配置列表。第 3 棒之后，每种配置都必须能跑。
MATRIX: dict[str, list[dict]] = {
    "E1": [
        {"run_id": "E1_dense_top5", "retriever": "dense", "topk": 5, "rerank": False},
    ],
    "E2": [
        {"run_id": "E2_bm25", "retriever": "bm25", "topk": 5, "rerank": False},
        {"run_id": "E2_rrf_alpha0.0", "retriever": "rrf", "alpha": 0.0, "topk": 5},
        {"run_id": "E2_rrf_alpha0.2", "retriever": "rrf", "alpha": 0.2, "topk": 5},
        {"run_id": "E2_rrf_alpha0.4", "retriever": "rrf", "alpha": 0.4, "topk": 5},
        {"run_id": "E2_rrf_alpha0.5", "retriever": "rrf", "alpha": 0.5, "topk": 5},
        {"run_id": "E2_rrf_alpha0.6", "retriever": "rrf", "alpha": 0.6, "topk": 5},
        {"run_id": "E2_rrf_alpha0.8", "retriever": "rrf", "alpha": 0.8, "topk": 5},
        {"run_id": "E2_rrf_alpha1.0", "retriever": "rrf", "alpha": 1.0, "topk": 5},
    ],
    "E3": [
        {"run_id": "E3_chunk_para", "retriever": "dense", "chunk": "paragraph", "topk": 5},
        {"run_id": "E3_chunk_256", "retriever": "dense", "chunk": 256, "topk": 5},
        {"run_id": "E3_chunk_512", "retriever": "dense", "chunk": 512, "overlap": 64, "topk": 5},
    ],
    "E4": [
        {"run_id": "E4_rerank_off", "retriever": "rrf", "rerank": False, "topk": 5},
        {"run_id": "E4_rerank_d20", "retriever": "rrf", "rerank": True, "depth": 20, "topk": 5},
        {"run_id": "E4_rerank_d50", "retriever": "rrf", "rerank": True, "depth": 50, "topk": 5},
        {"run_id": "E4_rerank_d100", "retriever": "rrf", "rerank": True, "depth": 100, "topk": 5},
    ],
    "E5": [
        {"run_id": "E5_pos_natural", "retriever": "rrf", "rerank": True, "order": "natural"},
        {"run_id": "E5_pos_gold_first", "retriever": "rrf", "rerank": True, "order": "gold_first"},
        {"run_id": "E5_pos_gold_middle", "retriever": "rrf", "rerank": True, "order": "gold_middle"},
        {"run_id": "E5_pos_gold_last", "retriever": "rrf", "rerank": True, "order": "gold_last"},
    ],
    "E6": [
        {"run_id": "E6_rm3", "retriever": "rrf", "query_rewrite": "rm3", "topk": 5},
    ],
}

# 第 3 棒的"配置矩阵可切换"最小验收集
SMOKE_CONFIGS = [
    {"run_id": "smoke_dense", "retriever": "dense"},
    {"run_id": "smoke_bm25", "retriever": "bm25"},
    {"run_id": "smoke_rrf_a0.5", "retriever": "rrf", "alpha": 0.5},
    {"run_id": "smoke_rrf_a1.0", "retriever": "rrf", "alpha": 1.0},
    {"run_id": "smoke_dense_rerank", "retriever": "dense", "rerank": True},
    {"run_id": "smoke_rrf_rerank", "retriever": "rrf", "alpha": 0.5, "rerank": True},
]


def cmd_list() -> None:
    total = sum(len(v) for v in MATRIX.values())
    print(f"实验配置矩阵（共 {len(MATRIX)} 组 / {total} 种配置）")
    print("=" * 74)
    for exp, cfgs in MATRIX.items():
        print(f"\n{exp}  ({len(cfgs)} 种)")
        for cfg in cfgs:
            extra = ", ".join(f"{k}={v}" for k, v in cfg.items() if k != "run_id")
            print(f"  - {cfg['run_id']:22s} {extra}")
    print("\n" + "=" * 74)
    print("smoke 最小验收集（第 3 棒）：", ", ".join(c["run_id"] for c in SMOKE_CONFIGS))


def cmd_dry(exp: str) -> int:
    configs = SMOKE_CONFIGS if exp == "smoke" else MATRIX.get(exp)
    if not configs:
        print(f"[X] 没有 {exp} 这组配置，用 --list 查看", file=sys.stderr)
        return 2

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[{exp}] 配置矩阵 dry-run（假检索结果 + 真实指标代码）")
    print("-" * 74)
    print(f"{'run_id':24s}{'recall@5':>10s}{'mrr':>8s}{'f1':>8s}  {'状态':<6s}")
    print("-" * 74)

    ok = 0
    for cfg in configs:
        res = run_dry(run_id=cfg["run_id"], topk=cfg.get("topk", 5))
        # 把配置写回结果，证明"配置可切换"
        res["config"].update({k: v for k, v in cfg.items() if k != "run_id"})
        out = RAW_DIR / f"{cfg['run_id']}.json"
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        m = res["metrics"]
        print(f"{cfg['run_id']:24s}{m['recall@5']:10.3f}{m['mrr']:8.3f}{m['f1']:8.3f}  OK")
        ok += 1

    print("-" * 74)
    print(f"[OK] {ok}/{len(configs)} 种配置全部跑通 → results/raw/{exp}_*.json")
    print("     注意：dry-run 的数字是假的（金标排名是模拟的），只验证接线。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="批量实验入口")
    ap.add_argument("--list", action="store_true", help="打印配置矩阵")
    ap.add_argument("--exp", default=None, help="E1..E6 或 smoke")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--split", default="smoke", choices=["smoke", "full"])
    args = ap.parse_args()

    if args.list or args.exp is None:
        cmd_list()
        return 0
    if args.dry_run:
        return cmd_dry(args.exp)

    print(f"[!] 真实运行尚未接线（第 3 棒的活）。\n"
          f"    A：把 run_exp 的分支接到 run_e1.run_real()，按 config 组装\n"
          f"       retriever / fusion / rerank / context 各组件。\n"
          f"    现在可以用：python eval/run_exp.py --exp {args.exp} --dry-run",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())