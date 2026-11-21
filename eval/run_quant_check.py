"""int8 量化 vs fp32 权重：检索指标与编码吞吐对比（B 的补充实验）。

背景：正式实验全程用 ``model_int8.onnx``（fp32 权重当时下不动）。补到 fp32 之后
需要回答："量化到底让检索掉了多少？" 于是对同一份数据、同一套流程分别用
fp32 / int8 编码，比较：

1. 单路稠密检索（dense）的 Recall@5 / MRR / nDCG@5
2. RRF 融合（α=0.2，E2 最优）的同上指标
3. 文档侧 / 查询侧的编码吞吐（条/秒）
4. 两套向量的一致性（同一个 doc 的余弦相似度均值）

输出：``results/tab/E2_quant.csv`` + ``results/raw/E2_quant.json``

用法：
    python eval/run_quant_check.py [--split full] [--alpha 0.2]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import eval.io_utils as IU  # noqa: E402
from eval import retrieval_eval as RE  # noqa: E402
from retrieval import BM25, BgeEncoder, FaissIndex, fuse_ids  # noqa: E402
from retrieval.rrf import alpha_weights  # noqa: E402

MODELS = {
    "fp32": "models/bge-small-zh-v1.5-onnx/model.onnx",
    "int8": "models/bge-small-zh-v1.5-onnx/model_int8.onnx",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="full", choices=["smoke", "full"])
    ap.add_argument("--alpha", type=float, default=0.2)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = IU.load_config()
    docs, queries = IU.load_split(args.split, cfg)
    questions = [q["question"] for q in queries]
    golds = [q["gold_doc_id"] for q in queries]
    print(f"[data] docs={len(docs)} queries={len(questions)}", flush=True)

    bm25 = BM25(k1=float(cfg["retrieval"]["sparse"]["k1"]), b=float(cfg["retrieval"]["sparse"]["b"]))
    bm25.fit(docs)
    sparse_ids = [[d for d, _ in bm25.search(q, topk=100)] for q in questions]
    fusion = cfg.get("retrieval", {}).get("fusion", {})
    fuse_k = int(fusion.get("k", 60))          # 超参一律走 config.yaml
    w = alpha_weights(args.alpha)

    rows, detail = [], {}
    embs_by_name = {}
    for name, rel in MODELS.items():
        path = root / rel
        if not path.exists():
            print(f"[skip] {name}: {rel} 不存在", flush=True)
            continue
        enc = BgeEncoder(model_path=str(path), backend="onnx")
        # 编码耗时必须"真编码"才有意义：build_from_encoder 会命中磁盘缓存（0.1–0.3 s），
        # 那样记下来的吞吐是缓存的吞吐，不是模型的速度。这里改成直接编码全库、再手工建索引。
        t0 = time.time()
        embs = np.asarray(enc.encode_doc_texts([d["text"] for d in docs], show_progress=False),
                          dtype=np.float32)
        enc_doc_s = time.time() - t0
        idx = FaissIndex(use_faiss=False).build([d["doc_id"] for d in docs], embs)
        embs_by_name[name] = idx._embs

        t0 = time.time()
        q_embs = np.asarray(enc.encode_queries(questions, show_progress=False), dtype=np.float32)
        enc_q_s = time.time() - t0

        dense_ids = idx.search_ids_batch(q_embs, 100)
        dense_m = RE.metrics_dict([[d for d in run] for run in dense_ids], golds)
        rrf = [fuse_ids([sparse_ids[i], dense_ids[i]], weights=w, k=fuse_k) for i in range(len(questions))]
        rrf_m = RE.metrics_dict(rrf, golds)
        rate = len(docs) / enc_doc_s if enc_doc_s > 0 else 0.0
        rows.append({
            "model": Path(str(path)).name,
            "backend": name,
            "dense_recall@5": round(dense_m["recall@5"], 6),
            "dense_mrr": round(dense_m["mrr"], 6),
            "rrf_alpha": args.alpha,
            "rrf_k": fuse_k,
            "rrf_recall@5": round(rrf_m["recall@5"], 6),
            "rrf_mrr": round(rrf_m["mrr"], 6),
            "rrf_ndcg@5": round(rrf_m["ndcg@5"], 6),
            "encode_docs_s": round(enc_doc_s, 1),
            "encode_queries_s": round(enc_q_s, 1),
            "docs_per_s": round(rate, 1),
        })
        detail[name] = {
            "dense": {k: round(v, 6) for k, v in dense_m.items() if isinstance(v, (int, float))},
            "rrf": {k: round(v, 6) for k, v in rrf_m.items() if isinstance(v, (int, float))},
            "encode_docs_s": round(enc_doc_s, 1),
            "encode_queries_s": round(enc_q_s, 1),
            "docs_per_s": round(rate, 1),
        }
        print(f"[{name}] dense R@5={dense_m['recall@5']:.4f} MRR={dense_m['mrr']:.4f} | "
              f"RRF R@5={rrf_m['recall@5']:.4f} MRR={rrf_m['mrr']:.4f} | {rate:.0f} docs/s",
              flush=True)

    if len(embs_by_name) == 2:
        a, b = embs_by_name["fp32"], embs_by_name["int8"]
        cos = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
        detail["vector_agreement"] = {
            "cosine_mean": round(float(cos.mean()), 6),
            "cosine_min": round(float(cos.min()), 6),
            "cosine_p1": round(float(np.percentile(cos, 1)), 6),
        }
        print(f"[agreement] 同一段落两套向量的余弦：mean={cos.mean():.4f} "
              f"min={cos.min():.4f} p1={np.percentile(cos, 1):.4f}", flush=True)

    IU.write_table("E2_quant", rows, cfg)
    IU.dump_json({"split": args.split, "alpha": args.alpha, "rows": rows, "detail": detail},
                 IU.path_of(cfg, "results_raw") / "E2_quant.json")
    print("[done] results/tab/E2_quant.csv", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
