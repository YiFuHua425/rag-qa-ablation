#!/usr/bin/env python
"""画图：E2 α 曲线、E3 切块曲线 —— 负责人：B

    python eval/plots.py                 # 读 results/raw/*_summary.json，出全部图
    python eval/plots.py --only e2

产出（文件名与 README 的实验口径表严格一致，D 直接引用）::

    results/fig/E2_alpha_recall.png   Recall@5 vs α（带 95% bootstrap CI 阴影）+ MRR/nDCG 副图
    results/fig/E3_chunk.png          Recall@5 柱状（带 CI）+ 索引膨胀 vs 上下文开销 散点
    results/fig/E2_latency.png        α 与延迟的关系（CPU，单查询）

中文字体缺失时自动退回英文标签，保证任何机器都能出图（不会画成豆腐块）。
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

_CJK_CANDIDATES = ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC",
                   "WenQuanYi Zen Hei", "Source Han Sans SC"]
_CJK_OK = False


def _setup_font() -> bool:
    global _CJK_OK
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CJK_CANDIDATES:
        if name in installed:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            _CJK_OK = True
            return True
    _CJK_OK = False
    return False


def T(zh: str, en: str) -> str:
    return zh if _CJK_OK else en


def _load(name: str) -> dict | None:
    path = ROOT / "results" / "raw" / name
    if not path.exists():
        print(f"[skip] 缺 {path.name}，先跑对应实验")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def plot_e2(summary: dict, cfg: dict) -> Path | None:
    rows = [r for r in summary.get("rows", []) if r.get("retriever") in ("rrf", "bm25", "dense")]
    rrf = sorted([r for r in rows if r.get("alpha") is not None and r.get("retriever") == "rrf"],
                 key=lambda r: r["alpha"])
    if not rrf:
        print("[skip] E2 汇总里没有 α 行")
        return None
    base = {r["retriever"]: r for r in rows if r.get("retriever") in ("bm25", "dense")}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    alphas = [r["alpha"] for r in rrf]
    r5 = [r["recall@5"] for r in rrf]
    lo = [(r.get("recall@5_ci") or [None, None])[0] for r in rrf]
    hi = [(r.get("recall@5_ci") or [None, None])[1] for r in rrf]
    ax.plot(alphas, r5, "o-", color="#1f77b4", lw=2, label=T("RRF 融合 (BM25+稠密)", "RRF fuse"))
    if all(v is not None for v in lo + hi):
        ax.fill_between(alphas, lo, hi, color="#1f77b4", alpha=0.18,
                        label=T("95% bootstrap CI", "95% CI"))
    if "bm25" in base:
        ax.axhline(base["bm25"]["recall@5"], ls="--", color="#d62728", lw=1.4,
                   label=T(f"纯 BM25 ({base['bm25']['recall@5']:.3f})", "BM25 only"))
    if "dense" in base:
        ax.axhline(base["dense"]["recall@5"], ls=":", color="#2ca02c", lw=1.6,
                   label=T(f"纯稠密 BGE ({base['dense']['recall@5']:.3f})", "Dense only"))
    best = max(rrf, key=lambda r: r["recall@5"])
    ax.scatter([best["alpha"]], [best["recall@5"]], s=110, facecolors="none",
               edgecolors="#ff7f0e", linewidths=2, zorder=5,
               label=T(f"最佳 α={best['alpha']:.1f}", f"best α={best['alpha']:.1f}"))
    ax.set_xlabel(T("稠密路权重 α（0=纯 BM25，1=纯稠密）", "alpha (0=BM25, 1=dense)"))
    ax.set_ylabel("Recall@5")
    ax.set_title(T("E2：混合检索 α 扫描（段落级 Recall@5）", "E2: alpha sweep, Recall@5"))
    ax.set_xlim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower center")

    ax = axes[1]
    ax.plot(alphas, [r["mrr"] for r in rrf], "s-", color="#9467bd", lw=2, label="MRR")
    ax.plot(alphas, [r["ndcg@5"] for r in rrf], "^-", color="#8c564b", lw=2, label="nDCG@5")
    ax.plot(alphas, [r["recall@1"] for r in rrf], "d-", color="#17becf", lw=1.6, label="Recall@1")
    ax.set_xlabel(T("稠密路权重 α", "alpha"))
    ax.set_ylabel(T("指标值", "score"))
    ax.set_title(T("E2：排序质量（MRR / nDCG@5 / Recall@1）", "E2: ranking quality"))
    ax.set_xlim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle(T(f"E2 混合检索消融（{summary.get('split', 'full')} split，RRF k="
                   f"{summary.get('config', {}).get('rrf_k', 60)}）",
                   "E2 hybrid retrieval ablation"), fontsize=11)
    fig.tight_layout()
    out = path_of(cfg, "results_fig") / "E2_alpha_recall.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}")
    return out


def plot_e2_latency(summary: dict, cfg: dict) -> Path | None:
    rows = [r for r in summary.get("rows", []) if r.get("alpha") is not None
            and r.get("latency_ms") is not None]
    if not rows:
        return None
    rows.sort(key=lambda r: r["alpha"])
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot([r["alpha"] for r in rows], [r["latency_ms"] for r in rows], "o-", color="#ff7f0e")
    ax.set_xlabel(T("稠密路权重 α", "alpha"))
    ax.set_ylabel(T("单查询耗时 (ms)", "latency (ms)"))
    ax.set_title(T("E2：α 对检索延迟的影响（CPU）", "E2: latency vs alpha (CPU)"))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = path_of(cfg, "results_fig") / "E2_latency.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}")
    return out


def plot_e3(summary: dict, cfg: dict) -> Path | None:
    rows = summary.get("rows", [])
    if not rows:
        print("[skip] E3 汇总为空")
        return None
    labels_zh = {"paragraph": "整段", "sentence": "句级", "128": "128字", "256": "256字",
                 "256o64": "256+64", "512": "512+64", "512o128": "512+128"}
    rows = sorted(rows, key=lambda r: r.get("chunks_per_doc", 0))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    ax = axes[0]
    xs = range(len(rows))
    vals = [r["recall@5"] for r in rows]
    err_lo = [r["recall@5"] - r["recall@5_ci"][0] for r in rows]
    err_hi = [r["recall@5_ci"][1] - r["recall@5"] for r in rows]
    bars = ax.bar(list(xs), vals, yerr=[err_lo, err_hi], capsize=4, color="#4c72b0", alpha=0.85)
    for rect, r in zip(bars, rows):
        top = r["recall@5_ci"][1] + 0.0025
        ax.text(rect.get_x() + rect.get_width() / 2, top,
                f"{r['recall@5']:.3f}", ha="center", va="bottom", fontsize=8.5, color="#1f2a37")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([labels_zh.get(r["strategy"], r["strategy"]) for r in rows], fontsize=9)
    ax.set_ylabel("Recall@5")
    ax.set_ylim(min(vals) - 0.03,
                min(1.0, max(r["recall@5_ci"][1] for r in rows) + 0.022))
    ax.set_title(T("E3：切块策略对段落召回的影响", "E3: chunking vs Recall@5"))
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    for r in rows:
        ax.scatter(r["chunks_per_doc"], r["context_chars@5"], s=70, alpha=0.85)
        ax.annotate(labels_zh.get(r["strategy"], r["strategy"]),
                    (r["chunks_per_doc"], r["context_chars@5"]),
                    textcoords="offset points", xytext=(6, -3), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel(T("索引膨胀：每段切成几块（对数轴）", "chunks per doc (log)"))
    ax.set_ylabel(T("top-5 上下文平均字数", "context chars@5"))
    ax.set_title(T("E3：切块越细，召回越高、上下文越省", "E3: granularity trade-off"))
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    out = path_of(cfg, "results_fig") / "E3_chunk.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}")
    return out


def plot_e6(cfg: dict) -> Path | None:
    path = ROOT / "results" / "raw" / "E6_summary.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("rows", [])
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    names = [r.get("label", r["run_id"]) for r in rows]
    vals = [r.get("recall@5", 0) for r in rows]
    bars = ax.bar(range(len(rows)), vals, color=["#8c8c8c", "#4c72b0"][: len(rows)], alpha=0.9)
    for rect, v in zip(bars, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.002, f"{v:.4f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(min(vals) - 0.02, min(1.0, max(vals) + 0.02))
    ax.set_ylabel("Recall@5")
    ax.set_title(T("E6：RM3 伪相关反馈前后对比", "E6: baseline vs RM3"))
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = path_of(cfg, "results_fig") / "E6_rm3.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="出图：E2 α 曲线 / E3 切块曲线 / E6 RM3")
    ap.add_argument("--only", default="", help="e2 / e3 / e6，逗号分隔；缺省全出")
    args = ap.parse_args()
    cfg = load_config()
    _setup_font()
    print(f"[font] {'中文 OK' if _CJK_OK else '未找到中文字体，图内文字用英文'}")
    which = [w.strip() for w in args.only.split(",") if w.strip()] or ["e2", "e3", "e6"]

    if "e2" in which:
        s2 = _load("E2_summary.json")
        if s2:
            plot_e2(s2, cfg)
            plot_e2_latency(s2, cfg)
    if "e3" in which:
        s3 = _load("E3_summary.json")
        if s3:
            plot_e3(s3, cfg)
    if "e6" in which:
        plot_e6(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
