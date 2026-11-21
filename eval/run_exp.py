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


# --------------------------------------------------------------- 真实运行（第 3 棒接线）
#
# 分工 B 把「一个入口、多套配置」接到真实检索链路上：retriever=dense|bm25|rrf(hybrid)
# 与 chunk 档位都由上面的 MATRIX 驱动，产出格式与 run_e2/run_e3 逐字对齐。
# 依赖 C 的重排 / 位置偏差、以及 RM3 的配置会被跳过并提示对应脚本。


def _lazy() -> dict:
    """真实运行才 import 检索链路，保证 --list / --dry-run 仍然零依赖。"""
    from eval import retrieval_eval as RE
    from eval.io_utils import Logger, load_config, load_split, write_run
    from retrieval.bge_encoder import DEFAULT_CACHE_DIR, BgeEncoder, FaissIndex
    from retrieval.bm25 import BM25
    from retrieval.chunking import build_chunks, chunks_to_doc_ranking
    from retrieval.rrf import fuse
    from retrieval.tokenization import build_tokenizer
    return {
        "RE": RE, "Logger": Logger, "load_config": load_config, "load_split": load_split,
        "write_run": write_run, "DEFAULT_CACHE_DIR": DEFAULT_CACHE_DIR,
        "BgeEncoder": BgeEncoder, "FaissIndex": FaissIndex, "BM25": BM25,
        "build_chunks": build_chunks, "chunks_to_doc_ranking": chunks_to_doc_ranking,
        "fuse": fuse, "build_tokenizer": build_tokenizer,
    }


def _chunk_key(cfg_row: dict, strategies: dict | None = None) -> str:
    """MATRIX 里的 chunk/overlap 两个字段 → run_e3.STRATEGIES 的键。

    MATRIX 的 ``{"chunk": 512, "overlap": 64}`` 与 run_e3 的 ``"512"``（size=512, overlap=64）
    是同一档，所以先试 ``512o64``，不存在再退 ``512``。
    """
    raw = cfg_row.get("chunk")
    if raw in (None, "paragraph"):
        return "paragraph"
    if strategies is None:
        from eval.run_e3 import STRATEGIES as strategies  # 惰性：保持 --list/--dry-run 零依赖
    size = int(raw)
    overlap = cfg_row.get("overlap")
    for key in (f"{size}o{overlap}", str(size)):
        if key in strategies:
            return key
    raise SystemExit(f"未知切块档位 chunk={raw} overlap={overlap}，可选 {sorted(strategies)}")


class _Lab:
    """一次运行内复用：语料、查询向量、BM25、doc/块索引。"""

    def __init__(self, mods: dict, cfg: dict, split: str, backend: str,
                 model_path: str | None, limit_queries: int):
        from eval.run_e3 import STRATEGIES

        self.m, self.STRATEGIES, self.cfg = mods, STRATEGIES, cfg
        docs, queries = mods["load_split"](split, cfg)
        if limit_queries:
            queries = queries[:limit_queries]
        self.docs = docs
        self.questions = [q["question"] for q in queries]
        self.golds = [q["gold_doc_id"] for q in queries]
        self.qids = [q["qid"] for q in queries]
        self.doc_ids = [d["doc_id"] for d in docs]
        self.doc_texts = [d["text"] for d in docs]

        dense = cfg.get("retrieval", {}).get("dense", {})
        self.encoder = mods["BgeEncoder"](
            model_name=dense.get("model", "BAAI/bge-small-zh-v1.5"),
            query_prefix=dense.get("query_prefix", ""),
            normalize=bool(dense.get("normalize", True)),
            batch_size=int(dense.get("encode_batch", 64)),
            model_path=model_path, backend=backend,
        )
        fusion = cfg.get("retrieval", {}).get("fusion", {})
        self.depth = int(fusion.get("dense_depth", 100))
        self.fuse_k = int(fusion.get("k", 60))
        self.sparse_topk = int(cfg.get("retrieval", {}).get("sparse", {}).get("topk", 100))
        self.topk = int(cfg.get("eval", {}).get("topk", 5))
        self._qemb = None
        self._cache: dict[str, dict] = {}

    @property
    def q_emb(self):
        if self._qemb is None:
            self._qemb = self.encoder.encode_queries(self.questions, show_progress=True)
        return self._qemb

    def _tokenizer(self):
        name = self.cfg.get("retrieval", {}).get("sparse", {}).get("tokenizer", "jieba")
        return self.m["build_tokenizer"](None if name in ("auto", "jieba") else name)

    def unit(self, key: str) -> dict:
        """检索单元：整段或某个切块档；返回 ids / texts / chunks / bm25 / index / dense。"""
        if key in self._cache:
            return self._cache[key]
        if key == "paragraph":
            ids, texts, chunks = self.doc_ids, self.doc_texts, None
        else:
            strategy, _ = self.STRATEGIES[key]
            chunks = self.m["build_chunks"](self.doc_texts, strategy)
            ids = [str(c.chunk_id) for c in chunks]
            texts = [c.text for c in chunks]
        sparse = self.cfg.get("retrieval", {}).get("sparse", {})
        bm25 = self.m["BM25"](k1=float(sparse.get("k1", 1.5)),
                              b=float(sparse.get("b", 0.75)), tokenizer=self._tokenizer())
        bm25.fit([{"doc_id": i, "text": t} for i, t in zip(ids, texts)])
        index = self.m["FaissIndex"]().build_from_encoder(
            self.encoder, [{"doc_id": i, "text": t} for i, t in zip(ids, texts)],
            cache_dir=self.m["DEFAULT_CACHE_DIR"], tag=f"runexp_{key}",
        )
        node = {"ids": ids, "texts": texts, "chunks": chunks, "bm25": bm25, "index": index,
                "dense": index.search_ids_batch(self.q_emb, topk=self.depth)}
        self._cache[key] = node
        return node

    def rank(self, cfg_row: dict):
        """单条配置 → 段落级排名列表（与 E2 同口径）。"""
        key = _chunk_key(cfg_row, self.STRATEGIES)
        node = self.unit(key)
        mode = cfg_row.get("retriever", "dense")
        if mode == "hybrid":
            mode = "rrf"
        alpha = float(cfg_row.get("alpha", 0.0)) if mode == "rrf" else None

        rankings = []
        for i, q in enumerate(self.questions):
            if mode == "bm25":
                fused = [(d, 0.0) for d in node["bm25"].topk_ids(q, topk=self.depth)]
            elif mode == "dense":
                fused = [(d, 0.0) for d in node["dense"][i]]
            else:
                sparse = node["bm25"].topk_ids(q, topk=self.depth)
                fused = self.m["fuse"]([[(d, 0.0) for d in sparse],
                                         [(d, 0.0) for d in node["dense"][i]]],
                                       weights=[1.0 - alpha, alpha], k=self.fuse_k)
            top = fused[: max(self.topk, 5)]
            if node["chunks"] is None:
                # 整段档：BM25 与稠密索引的 key 都是 doc_id，直接就是段落排名
                rankings.append([d for d, _ in top][: self.topk])
            else:
                # 切块档：chunk_id 是 chunks 列表下标（字符串形式存进索引），压平成段落排名
                doc_rows = self.m["chunks_to_doc_ranking"]([int(c) for c, _ in top],
                                                           node["chunks"])
                rankings.append([self.doc_ids[d] for d in doc_rows][: self.topk])
        return rankings, {"mode": mode, "alpha": alpha, "chunk": key, "node": node}

    def probe_ms(self, mode: str, alpha: float | None, key: str) -> float:
        """单查询端到端（编码 + 打分 + 融合），30 题 × 3 轮取中位数。"""
        import time as _t

        node = self.unit(key)
        times = []
        for _ in range(3):
            for q in self.questions[: min(30, len(self.questions))]:
                t0 = _t.perf_counter()
                if mode == "bm25":
                    node["bm25"].topk_ids(q, topk=self.topk)
                else:
                    node["index"].search_ids(self.encoder.encode_query(q), self.depth)
                    if mode == "rrf":
                        sparse = node["bm25"].topk_ids(q, topk=self.sparse_topk)
                        self.m["fuse"]([[(d, 0.0) for d in sparse],
                                        [(d, 0.0) for d in node["dense"][0]]],
                                       weights=[1.0 - (alpha or 0.0), alpha or 0.0], k=self.fuse_k)
                times.append((_t.perf_counter() - t0) * 1000)
        times.sort()
        return round(times[len(times) // 2], 3)


def cmd_real(exp: str, split: str = "smoke", backend: str = "auto",
             model_path: str | None = None, limit_queries: int = 0) -> int:
    configs = SMOKE_CONFIGS if exp == "smoke" else MATRIX.get(exp)
    if not configs:
        print(f"[X] 没有 {exp} 这组配置，用 --list 查看", file=sys.stderr)
        return 2

    mods = _lazy()
    cfg = mods["load_config"]()
    RE = mods["RE"]
    log = mods["Logger"](f"run_exp_{exp}", cfg)
    log(f"[run_exp] exp={exp} split={split} configs={len(configs)} encoder={backend}")

    lab = _Lab(mods, cfg, split, backend, model_path, limit_queries)
    log(f"[data] n_docs={len(lab.docs)} n_questions={len(lab.questions)} | "
        f"encoder={lab.encoder.model_name} backend={lab.encoder._engine}")
    boot_n = int(cfg.get("eval", {}).get("bootstrap_n", 1000))
    boot_alpha = float(cfg.get("eval", {}).get("alpha", 0.05))
    seed = int(cfg.get("project", {}).get("seed", 42))

    print("-" * 78)
    print(f"{'run_id':24s}{'recall@5':>10s}{'mrr':>8s}{'ndcg@5':>9s}{'lat(ms)':>9s}  状态")
    print("-" * 78)

    ok = skipped = 0
    for cfg_row in configs:
        run_id = cfg_row["run_id"]
        # 重排 / 位置偏差 / 查询改写不在检索链路上：交给 C 的 rerank 与 B 的 run_e6_rm3.py
        if cfg_row.get("rerank") or cfg_row.get("order") or cfg_row.get("query_rewrite"):
            why = ("rerank" if cfg_row.get("rerank") else
                   ("position" if cfg_row.get("order") else "query_rewrite"))
            print(f"{run_id:24s}{'-':>10s}{'-':>8s}{'-':>9s}{'-':>9s}  skip（{why} 见对应脚本）")
            skipped += 1
            continue

        rankings, extra = lab.rank(cfg_row)
        met = RE.metrics_dict(rankings, lab.golds, ks=(1, 5, 10, 20), ndcg_k=5)
        met["recall@5_ci"] = RE.recall_ci(rankings, lab.golds, 5, boot_n, boot_alpha, seed)
        met["latency_ms"] = lab.probe_ms(extra["mode"], extra["alpha"], extra["chunk"])
        path = mods["write_run"](
            run_id,
            config={"retriever": extra["mode"], "alpha": extra["alpha"],
                    "chunk": extra["chunk"], "topk": lab.topk, "depth": lab.depth,
                    "sparse_topk": lab.sparse_topk, "encoder": backend},
            metrics=met,
            per_question=RE.per_question_records(lab.qids, rankings, lab.golds),
            meta={"split": split, "n_questions": len(lab.questions)},
            cfg=cfg,
        )
        print(f"{run_id:24s}{met['recall@5']:10.4f}{met['mrr']:8.4f}{met['ndcg@5']:9.4f}"
              f"{met['latency_ms']:9.2f}  OK → {path.name}")
        ok += 1

    print("-" * 78)
    print(f"[OK] {ok}/{len(configs)} 种配置跑完"
          + (f"，{skipped} 种跳过（重排/位置/RM3 走各自脚本）" if skipped else ""))
    log(f"[done] {ok} ok / {skipped} skipped")
    log.close()
    return 0 if ok else 2


def main() -> int:
    ap = argparse.ArgumentParser(description="批量实验入口")
    ap.add_argument("--list", action="store_true", help="打印配置矩阵")
    ap.add_argument("--exp", default=None, help="E1..E6 或 smoke")
    ap.add_argument("--dry-run", action="store_true", help="不碰模型：假检索结果 + 真指标代码")
    ap.add_argument("--split", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--encoder", default="auto", choices=["auto", "onnx", "st", "hash"])
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--limit-queries", type=int, default=0)
    args = ap.parse_args()

    if args.list or args.exp is None:
        cmd_list()
        return 0
    if args.dry_run:
        return cmd_dry(args.exp)
    return cmd_real(args.exp, args.split, args.encoder, args.model_path, args.limit_queries)


if __name__ == "__main__":
    raise SystemExit(main())