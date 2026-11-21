"""E2 的补充实验（超参敏感度），产出 ``results/tab/E2_extras.csv``

四项，都是"报告里能单独成段"的实用问题：

1. ``rrf_k``：RRF 平滑常数 k 取 10/20/60/100/200 对 Recall 的影响（k=60 是经验值）；
2. ``depth``：两路各取多少候选再融合（20/50/100/200）；
3. ``bm25_k1b``：BM25 的 k1/b 网格，看稀疏通道本身对超参是否敏感；
4. ``normalize``：稠密向量不归一化（退化为内积）时 Recall 掉多少。

用法::

    python eval/run_extras_b.py --split full
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from eval import io_utils as IU  # noqa: E402
from eval import retrieval_eval as RE  # noqa: E402
from retrieval.bge_encoder import BgeEncoder, FaissIndex  # noqa: E402
from retrieval.bm25 import BM25  # noqa: E402
from retrieval.rrf import fuse  # noqa: E402


def _prep(args):
    cfg = IU.load_config(args.config)
    docs, queries = IU.load_split(args.split, cfg)
    questions = [q["question"] for q in queries]
    golds = [q["gold_doc_id"] for q in queries]
    enc = BgeEncoder(model_name=cfg["retrieval"]["dense"]["model"],
                     model_path=args.model_path, backend=args.encoder,
                     batch_size=cfg["retrieval"]["dense"].get("encode_batch", 64))
    t0 = time.perf_counter()
    index = FaissIndex().build_from_encoder(enc, docs, tag="docs")
    print(f"[dense] 索引就绪 {time.perf_counter() - t0:.1f}s dim={index.dim}")
    q_embs = enc.encode_queries(questions)
    return cfg, docs, questions, golds, enc, index, q_embs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "config.yaml"))
    ap.add_argument("--split", default="full", choices=["smoke", "full"])
    ap.add_argument("--encoder", default="auto")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--alpha", type=float, default=0.2)
    args = ap.parse_args()

    cfg, docs, questions, golds, enc, index, q_embs = _prep(args)
    seed = cfg.get("project", {}).get("seed", 42)
    boot_n = cfg.get("eval", {}).get("bootstrap_n", 1000)
    fusion = cfg.get("retrieval", {}).get("fusion", {})
    fuse_k = int(fusion.get("k", 60))          # 超参一律走 config.yaml
    base_depth = int(fusion.get("dense_depth", 100))
    rows: List[Dict[str, object]] = []

    # ---------------------------------------------------------- 1/2. k 与 depth
    max_depth = 200
    bm25 = BM25(k1=cfg["retrieval"]["sparse"]["k1"], b=cfg["retrieval"]["sparse"]["b"]).fit(docs)
    sparse = [bm25.topk_ids(q, topk=max_depth) for q in questions]
    dense = [list(index.search_ids(e, max_depth)) for e in q_embs]

    def fused_rankings(depth: int, k: int, alpha: float, nq: int | None = None) -> List[List[str]]:
        out = []
        for i in range(len(sparse) if nq is None else nq):
            r = fuse([[(d, 0.0) for d in sparse[i][:depth]],
                      [(d, 0.0) for d in dense[i][:depth]]],
                     weights=[1.0 - alpha, alpha], k=k)
            out.append([d for d, _ in r])
        return out

    def record(kind: str, param: str, value: object, rankings: List[List[str]]) -> None:
        met = RE.metrics_dict(rankings, golds, ks=(1, 5, 10), ndcg_k=5)
        lo, hi = RE.recall_ci(rankings, golds, 5, boot_n, 0.05, seed)
        rows.append({"family": kind, "param": param, "value": value,
                     "recall@5": round(met["recall@5"], 6), "recall@5_lo": round(lo, 6),
                     "recall@5_hi": round(hi, 6), "recall@1": round(met["recall@1"], 6),
                     "mrr": round(met["mrr"], 6), "ndcg@5": round(met["ndcg@5"], 6)})
        print(f"[{kind:9s}] {param}={value} → R@5={met['recall@5']:.4f} "
              f"MRR={met['mrr']:.4f} nDCG@5={met['ndcg@5']:.4f}")

    print("[1/5] RRF k 扫描（depth=%d, alpha=%.2f）" % (base_depth, args.alpha))
    for k in (10, 20, 60, 100, 200):
        record("rrf_k", "k", k, fused_rankings(base_depth, k, args.alpha))

    print("[2/5] depth 扫描（k=%d, alpha=%.2f）" % (fuse_k, args.alpha))
    for depth in (20, 50, 100, 200):
        record("depth", "depth", depth, fused_rankings(depth, fuse_k, args.alpha))

    # ---------------------------------------------------------- 3. BM25 k1/b 网格
    print("[3/5] BM25 k1/b 网格（纯 BM25）")
    for k1 in (0.9, 1.2, 1.5, 2.0):
        for b in (0.3, 0.5, 0.75, 0.9):
            m = BM25(k1=k1, b=b).fit(docs)
            rk = [[d for d, _ in m.search(q, topk=10)] for q in questions]
            record("bm25_k1b", f"k1={k1},b={b}", round(k1 * 100 + b, 2), rk)

    # ---------------------------------------------------------- 4. 归一化 abl ation
    print("[4/5] 稠密向量归一化 ablation")
    enc_raw = BgeEncoder(model_name=cfg["retrieval"]["dense"]["model"],
                         model_path=args.model_path, backend=args.encoder,
                         normalize=False, batch_size=cfg["retrieval"]["dense"].get("encode_batch", 64))
    from retrieval.bge_encoder import DEFAULT_CACHE_DIR  # noqa: F401  仅用于确认缓存目录存在

    index_raw = FaissIndex().build_from_encoder(enc_raw, docs, tag="docs_nonorm")
    q_raw = enc_raw.encode_queries(questions)
    for tag, idx, qs in (("normalized", index, q_embs), ("non_normalized", index_raw, q_raw)):
        dense_rk = [list(idx.search_ids(e, 100)) for e in qs]
        rk = []
        for i in range(len(questions)):
            r = fuse([[(d, 0.0) for d in sparse[i][:100]],
                      [(d, 0.0) for d in dense_rk[i][:100]]],
                     weights=[1.0 - args.alpha, args.alpha], k=fuse_k)
            rk.append([d for d, _ in r])
        record("normalize", "dense", tag, rk)
        # 纯稠密也报一份，缩小混杂因素
        record("normalize", "dense_only", tag, dense_rk)

    # ---------------------------------------------------------- 5. 指令前缀 ablation
    print("[5/5] 查询指令前缀 ablation")
    instr = "为这个句子生成表示以用于检索相关文章："
    enc_pf = BgeEncoder(model_name=cfg["retrieval"]["dense"]["model"],
                        model_path=args.model_path, backend=args.encoder,
                        query_prefix=instr, batch_size=cfg["retrieval"]["dense"].get("encode_batch", 64))
    index_pf = FaissIndex().build_from_encoder(enc_pf, docs, tag="docs_prefix")
    q_pf = enc_pf.encode_queries(questions)
    for tag, idx, qs in (("no_prefix", index, q_embs), ("with_prefix", index_pf, q_pf)):
        dense_rk = [list(idx.search_ids(e, 100)) for e in qs]
        rk = []
        for i in range(len(questions)):
            r = fuse([[(d, 0.0) for d in sparse[i][:100]],
                      [(d, 0.0) for d in dense_rk[i][:100]]],
                     weights=[1.0 - args.alpha, args.alpha], k=fuse_k)
            rk.append([d for d, _ in r])
        record("prefix", "dense_only", tag, dense_rk)
        record("prefix", "rrf", tag, rk)

    out_csv = IU.path_of(cfg, "results_tab") / "E2_extras.csv"
    IU.write_table("E2_extras", rows, cfg)
    IU.dump_json({"split": args.split, "alpha": args.alpha, "rrf_k": fuse_k, "base_depth": base_depth,
                   "model_file": Path(str(enc.model_file)).name, "index_backend": index.backend, "rows": rows},
                 IU.path_of(cfg, "results_raw") / "E2_extras.json")
    print(f"[done] 表 → {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
