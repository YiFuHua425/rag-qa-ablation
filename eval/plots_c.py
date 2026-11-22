#!/usr/bin/env python
"""C 的两张图：E4 精度–延迟 Pareto、E5 位置偏差 —— 负责人：C

    python eval/plots_c.py              # 有哪个 summary 画哪个
    python eval/plots_c.py --only e4

产出（文件名与 README 的实验编号对齐，E 同学直接引用）::

    results/fig/E4_pareto.png     左：总延迟 vs F1（标注深度）；右：Recall@5 柱状（带 95% CI）
    results/fig/E5_position.png   左：F1 vs 金标位置（分 max_chars 两条线）；右：金标保留率

字体沿用 ``eval/plots.py`` 的探测逻辑：机器上没有中文字体就自动退英文标签，
绝不把方块字画进报告。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplcache"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eval.io_utils import load_config, path_of  # noqa: E402
from eval.plots import T, _setup_font  # noqa: E402

C_NATURAL = "#4c72b0"
C_OFF = "#7f7f7f"
C_ON = "#c44e52"
PALETTE = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]


def _load(name: str) -> dict | None:
    path = ROOT / "results" / "raw" / name
    if not path.exists():
        print(f"[skip] 缺 {path.name}，先跑对应实验")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig, name: str, cfg: dict) -> Path:
    out_dir = path_of(cfg, "results_fig")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


# ------------------------------------------------------------------ E4
def plot_e4(cfg: dict | None = None) -> Path | None:
    cfg = cfg or load_config()
    data = _load("E4_summary.json")
    if not data:
        return None
    rows = data["rows"]
    if not rows:
        return None
    has_f1 = all(r.get("f1") is not None for r in rows)
    _setup_font()

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    ax = axes[0]
    xs = [r["latency_ms"] for r in rows]
    if has_f1:
        ys = [r["f1"] for r in rows]
        ylabel = T("F1", "F1")
    else:                                  # 阅读器未参与时用 MRR（对排序最敏感，曲线才有形状）
        ys = [r["mrr"] for r in rows]
        ylabel = T("MRR（F1 待阅读器权重）", "MRR (F1 pending reader)")
    labels = ["off" if r["depth"] is None else f"d{r['depth']}" for r in rows]
    ax.plot(xs, ys, "-", color="#bbbbbb", lw=1.2, zorder=1)
    ax.scatter(xs, ys, s=70, c=[C_OFF if r["depth"] is None else C_ON for r in rows],
               zorder=2, edgecolor="white", linewidth=1.2)
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set_xlabel(T("总延迟 (ms/query)", "Latency (ms/query)"))
    ax.set_ylabel(ylabel)
    ax.set_title(T("精度–延迟 Pareto（重排深度）", "Accuracy vs latency (rerank depth)"))
    ax.grid(alpha=0.25)

    ax = axes[1]
    names = [r["run_id"].replace("E4_rerank_", "") for r in rows]
    r5 = [r["recall@5"] for r in rows]
    lo = [(r.get("recall@5_ci") or [0, 0])[0] for r in rows]
    hi = [(r.get("recall@5_ci") or [0, 0])[1] for r in rows]
    err = [[max(0.0, a - b) for a, b in zip(r5, lo)], [max(0.0, b - a) for a, b in zip(hi, r5)]]
    colors = [C_OFF if r["depth"] is None else C_ON for r in rows]
    ax.bar(names, r5, color=colors, width=0.6, yerr=err, capsize=4,
           error_kw={"elinewidth": 1, "ecolor": "#444444"})
    for i, v in enumerate(r5):
        ax.text(i, v + 0.004, f"{v:.4f}", ha="center", fontsize=8)
    base = rows[0]["recall@5"]
    ax.axhline(base, ls="--", lw=1, color=C_OFF)
    finite = [v for v in r5 if v == v]
    low = min(finite + [base]) - 0.02
    high = max(finite + [base]) + 0.02
    ax.set_ylim(max(0.0, low), min(1.0, high))
    ax.set_ylabel("Recall@5")
    ax.set_title(T("重排是否把深候选捞上来", "Recall@5 by arm (95% CI)"))
    ax.grid(axis="y", alpha=0.25)

    return _save(fig, "E4_pareto.png", cfg)


# ------------------------------------------------------------------ E5
def plot_e5(cfg: dict | None = None) -> Path | None:
    cfg = cfg or load_config()
    data = _load("E5_summary.json")
    if not data:
        return None
    rows = [r for r in data["rows"]]
    if not rows:
        return None
    _setup_font()
    orders = ["natural", "gold_first", "gold_middle", "gold_last"]
    zh = {"natural": "自然顺序", "gold_first": "第一段", "gold_middle": "正中间",
          "gold_last": "最后一段"}
    en = {"natural": "natural", "gold_first": "first", "gold_middle": "middle",
          "gold_last": "last"}

    metric = "f1"
    cond = "gold_kept_f1"
    chars = sorted({r["max_chars"] for r in rows})

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    has_f1 = any(r.get("f1") is not None for r in rows)
    ax = axes[0]
    if not has_f1:
        # 阅读器还没跑：左图直接画"金标保留率"，比画一条空坐标轴有用
        for ci, mc in enumerate(chars):
            sub = {r["order"]: r for r in rows if r["max_chars"] == mc}
            xs = [i for i, o in enumerate(orders) if o in sub]
            ys = [sub[o]["gold_kept_rate"] for o in orders if o in sub]
            ax.plot(xs, ys, "o-", color=PALETTE[ci % len(PALETTE)], lw=2,
                    label=T(f"max_chars={mc}", f"max_chars={mc}"))
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=8, color=PALETTE[ci % len(PALETTE)])
        ax.set_xticks(range(len(orders)))
        ax.set_xticklabels([T(zh[o], en[o]) for o in orders])
        ax.set_ylim(-0.05, 1.1)
        ax.set_ylabel(T("金标进入上下文的比例（F1 待阅读器权重）", "gold kept rate (F1 pending)"))
        ax.set_title(T("金标位置 → 是否还在上下文里", "Gold position vs kept rate"))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    else:
        for ci, mc in enumerate(chars):
            sub = {r["order"]: r for r in rows if r["max_chars"] == mc}
            xs = [i for i, o in enumerate(orders) if o in sub]
            ys = [sub[o][metric] if sub[o].get(metric) is not None else float("nan") for o in orders
                  if o in sub]
            ax.plot(xs, ys, "o-", color=PALETTE[ci % len(PALETTE)], lw=2,
                    label=T(f"max_chars={mc}（全体）", f"max_chars={mc} (all)"))
            ys2 = [sub[o][cond] if sub[o].get(cond) is not None else float("nan") for o in orders
                   if o in sub]
            ax.plot(xs, ys2, "s--", color=PALETTE[ci % len(PALETTE)], lw=1.2, alpha=0.75,
                    label=T(f"max_chars={mc}（金标已进上下文）",
                            f"max_chars={mc} (gold kept)"))
            for x, y in zip(xs, ys):
                if y == y:
                    ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                                xytext=(0, 8), ha="center", fontsize=8,
                                color=PALETTE[ci % len(PALETTE)])
        ax.set_xticks(range(len(orders)))
        ax.set_xticklabels([T(zh[o], en[o]) for o in orders])
        ax.set_ylabel("F1")
        ax.set_title(T("金标位置 → 抽取 F1", "Gold position vs F1"))
        ax.legend(fontsize=7.5, loc="lower left")
        ax.grid(alpha=0.25)

    ax = axes[1]
    width = 0.38
    for ci, mc in enumerate(chars):
        sub = {r["order"]: r for r in rows if r["max_chars"] == mc}
        xs = [i + (ci - (len(chars) - 1) / 2) * width for i in range(len(orders)) if orders[i] in sub]
        ys = [sub[o]["gold_kept_rate"] for o in orders if o in sub]
        ax.bar(xs, ys, width=width, color=PALETTE[ci % len(PALETTE)], alpha=0.85,
               label=T(f"max_chars={mc}", f"max_chars={mc}"))
    ax.set_xticks(range(len(orders)))
    ax.set_xticklabels([T(zh[o], en[o]) for o in orders])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(T("金标进入上下文的比例", "gold kept rate"))
    ax.set_title(T("截断把金标挡在门外的比例", "Gold kept rate (truncation)"))
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    return _save(fig, "E5_position.png", cfg)


def main() -> int:
    ap = argparse.ArgumentParser(description="画 E4 / E5 的图")
    ap.add_argument("--only", default="e4,e5")
    args = ap.parse_args()
    cfg = load_config()
    todo = {x.strip().lower() for x in args.only.split(",") if x.strip()}
    if "e4" in todo:
        p = plot_e4(cfg)
        print(f"[fig] {p.relative_to(ROOT)}" if p else "[skip] E4 没画出来")
    if "e5" in todo:
        p = plot_e5(cfg)
        print(f"[fig] {p.relative_to(ROOT)}" if p else "[skip] E5 没画出来")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
