"""C 的实验底座：候选集 → 重排 → 上下文 → 阅读器 —— 负责人：C

E4（重排与候选深度）和 E5（位置偏差）共用同一条链路，区别只在「动哪个旋钮」：

* E4 动 :class:`~rerank.cross_encoder.Reranker` 的 ``depth``（20 / 50 / 100）与开关；
* E5 动 :func:`~reader.context_builder.build_context` 的 ``order`` 与 ``max_chars``。

**检索侧不重复实现**：直接复用 B 的 ``BM25`` / ``BgeEncoder`` / ``FaissIndex`` / ``fuse``，
按 config.yaml 的 α 融出一份候选集，落盘缓存到 ``.cache/cand/``——
E4 与 E5 跑的是**同一份候选**，否则两个实验的差异会混入"检索抖动"。

为什么候选要缓存：smoke 500 题（或 full 3219 题）跑一次稠密编码 + BM25 + 融合约 20 s，
E4 有 4 个 arm、E5 有 8 个 arm，每题重复检索 12 次纯属浪费；缓存键里带 α / depth / 题数 /
编码器权重名，任何一项变了都会重算。
"""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from eval import metrics as M
from eval import retrieval_eval as RE
from eval.io_utils import _p, load_split, path_of

ROOT = Path(__file__).resolve().parents[1]
CAND_DIR = ROOT / ".cache" / "cand"
RERANK_DIR = ROOT / ".cache" / "rerank"


# ------------------------------------------------------------------ 数据结构
@dataclass
class Bundle:
    """一次实验用到的数据 + 候选集（全部按题序对齐）。"""

    split: str
    docs: List[dict]
    queries: List[dict]
    candidates: List[List[Tuple[str, float]]]
    doc_map: Dict[str, str]
    info: Dict[str, object] = field(default_factory=dict)

    @property
    def questions(self) -> List[str]:
        return [q["question"] for q in self.queries]

    @property
    def golds(self) -> List[str]:
        return [q["gold_doc_id"] for q in self.queries]

    @property
    def refs(self) -> List[str]:
        return [q["answer_text"] for q in self.queries]

    @property
    def qids(self) -> List[str]:
        return [q["qid"] for q in self.queries]

    def __len__(self) -> int:
        return len(self.queries)


# ------------------------------------------------------------------ 候选集
def _cache_key(split: str, alpha: float, depth: int, n: int, stamp: str) -> str:
    raw = f"{split}|{alpha:.2f}|{depth}|{n}|{stamp}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def build_candidates(cfg: dict, split: str = "smoke", alpha: float | None = None,
                     depth: int = 100, encoder_backend: str = "auto",
                     model_path: str | None = None, limit: int = 0,
                     use_cache: bool = True) -> Bundle:
    """BM25 + 稠密 → RRF 融合 → ``candidates[i] = [(doc_id, score), ...]``（降序）。"""
    from retrieval.bge_encoder import DEFAULT_CACHE_DIR, BgeEncoder, FaissIndex
    from retrieval.bm25 import BM25
    from retrieval.rrf import fuse
    from retrieval.tokenization import build_tokenizer

    alpha = float(alpha if alpha is not None else
                  _p(cfg, "retrieval.fusion.alpha", 0.5))
    fuse_k = int(_p(cfg, "retrieval.fusion.k", 60))
    dense_depth = max(int(depth), int(_p(cfg, "retrieval.fusion.dense_depth", 100)))
    sparse_topk = max(int(depth), int(_p(cfg, "retrieval.sparse.topk", 100)))

    docs, queries = load_split(split, cfg)
    if limit:
        queries = queries[: int(limit)]
    dense_cfg = _p(cfg, "retrieval.dense", {}) or {}
    enc = BgeEncoder(
        model_name=dense_cfg.get("model", "BAAI/bge-small-zh-v1.5"),
        query_prefix=dense_cfg.get("query_prefix", ""),
        normalize=bool(dense_cfg.get("normalize", True)),
        batch_size=int(dense_cfg.get("encode_batch", 64)),
        model_path=model_path, backend=encoder_backend,
    )
    stamp = f"{Path(str(enc.model_file)).name}-{dense_cfg.get('query_prefix', '')}"
    key = _cache_key(split, alpha, dense_depth, len(queries), stamp)
    cache_path = CAND_DIR / f"cand_{split}_a{alpha:.2f}_d{dense_depth}_{key}.json"
    if use_cache and cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if payload.get("n") == len(queries):
                print(f"[cand] 命中缓存 {cache_path.name}", flush=True)
                return Bundle(split=split, docs=docs, queries=queries,
                              candidates=[[(d, float(s)) for d, s in row]
                                          for row in payload["candidates"]],
                              doc_map={d["doc_id"]: d["text"] for d in docs},
                              info={**payload["info"], "cached": True})
        except Exception:
            pass

    t0 = time.time()
    sparse_cfg = _p(cfg, "retrieval.sparse", {}) or {}
    tok_name = sparse_cfg.get("tokenizer", "jieba")
    bm25 = BM25(k1=float(sparse_cfg.get("k1", 1.5)), b=float(sparse_cfg.get("b", 0.75)),
                tokenizer=build_tokenizer(None if tok_name in ("auto", "jieba") else tok_name))
    bm25.fit(docs)
    index = FaissIndex()
    index.build_from_encoder(enc, docs, cache_dir=DEFAULT_CACHE_DIR, tag="docs")
    questions = [q["question"] for q in queries]
    q_emb = enc.encode_queries(questions, show_progress=True)
    dense_runs = index.search_ids_batch(q_emb, topk=dense_depth)

    candidates: List[List[Tuple[str, float]]] = []
    for q, d_run in zip(questions, dense_runs):
        s_run = bm25.search(q, topk=sparse_topk)
        fused = fuse([s_run, [(d, 0.0) for d in d_run]],
                     weights=[1.0 - alpha, alpha], k=fuse_k)
        candidates.append([(d, float(s)) for d, s in fused[: max(depth, 5)]])
    elapsed = time.time() - t0

    # 检索延迟：10 题 × 3 轮取中位数（BM25 打分 + 稠密内积 + RRF），口径与 E2 一致
    probe_n = min(10, len(questions))
    times: List[float] = []
    for _ in range(3):
        for i in range(probe_n):
            t = time.perf_counter()
            s_run = bm25.search(questions[i], topk=sparse_topk)
            d_run = index.search(enc.encode_query(questions[i]), dense_depth)
            fuse([s_run, d_run], weights=[1.0 - alpha, alpha], k=fuse_k)
            times.append((time.perf_counter() - t) * 1000.0)
    times.sort()
    lat = round(times[len(times) // 2], 3)

    info = {
        "split": split, "alpha": alpha, "depth": depth, "rrf_k": fuse_k,
        "sparse_topk": sparse_topk, "n_questions": len(queries),
        "encoder": enc.model_name, "model_file": Path(str(enc.model_file)).name,
        "index_backend": index.backend, "build_seconds": round(elapsed, 2),
        "retrieval_latency_ms": lat, "cached": False,
    }
    print(f"[cand] α={alpha} depth={depth} 候选就绪：{len(candidates)} 题，"
          f"{elapsed:.1f}s（{index.backend}）", flush=True)
    if use_cache:
        CAND_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(
            {"n": len(queries), "info": info, "candidates": candidates},
            ensure_ascii=False), encoding="utf-8")
    return Bundle(split=split, docs=docs, queries=queries, candidates=candidates,
                  doc_map={d["doc_id"]: d["text"] for d in docs}, info=info)


def retrieval_latency(bundle: Bundle, n: int = 10, repeat: int = 3) -> float:
    """候选生成阶段的单查询延迟（BM25 + 稠密 + RRF，10 题 × 3 轮中位数）。"""
    return float(bundle.info.get("retrieval_latency_ms", 0.0))


# ------------------------------------------------------------------ 重排缓存
def rerank_pool(bundle: Bundle, reranker, pool: int = 100,
                use_cache: bool = True) -> List[List[Tuple[str, float]]]:
    """对每题前 ``pool`` 个候选**只打一次分**，E4 的三个 depth 与 E5 都从这份结果切片。

    为什么这样等价：``depth=k`` 的语义是"只在前 k 个候选里挑最好的 5 个"，
    所以对 pool 打完分后取 ``ranked[:k][:5]`` 就是 depth=k 的结果（分数与排序完全相同）。
    三个 arm 共用一次前向，**省 1.7 倍算力**，而且彻底避免了"不同 arm 各自调模型带来的抖动"。

    打分结果落盘到 ``.cache/rerank/``：交叉编码器是整条链路最贵的一段
    （278M 参数 × 每题最多 100 个候选），E5 复用 E4 的 pool 就不必再跑一遍。
    """
    model_name = getattr(reranker, "model_path", None) or getattr(reranker, "model_name", "?")
    key = hashlib.sha1(
        f"{pool}|{bundle.info.get('alpha')}|{len(bundle)}|{model_name}|"
        f"{getattr(reranker, 'max_length', 512)}|{bundle.split}|"
        f"{'-'.join(bundle.qids[:3])}|{'-'.join(bundle.qids[-3:])}".encode("utf-8")
    ).hexdigest()[:16]
    path = RERANK_DIR / f"pool_{pool}_{key}.json"

    if use_cache and path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("n") == len(bundle):
                print(f"[rerank] 命中缓存 {path.name}", flush=True)
                return [[(d, float(s)) for d, s in row] for row in payload["rows"]]
        except Exception:
            pass

    t0 = time.time()
    rows = []
    for i, (q, cand) in enumerate(zip(bundle.questions, bundle.candidates), start=1):
        ranked = reranker.score_candidates(q, cand[:pool])
        rows.append([[d, round(float(s), 6)] for d, s in ranked])
        if i % 20 == 0 or i == len(bundle):
            print(f"\r[rerank] {i}/{len(bundle)} 题，{time.time() - t0:.0f}s", end="", flush=True)
    print(f"\n[rerank] pool={pool} 完成，{time.time() - t0:.1f}s"
          f"（{time.time() - t0:.2f}s/题）", flush=True)

    if use_cache:
        RERANK_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"n": len(bundle), "pool": pool,
                                    "model": str(model_name),
                                    "alpha": bundle.info.get("alpha"),
                                    "rows": rows}, ensure_ascii=False), encoding="utf-8")
    return [[(d, float(s)) for d, s in row] for row in rows]


def topk_from_pool(pooled: Sequence[Sequence[Tuple[str, float]]], depth: int,
                   topn: int = 5) -> List[List[str]]:
    """``depth`` 语义：只在前 ``depth`` 个候选里取最好的 ``topn`` 个。"""
    return [[d for d, _ in row[:depth]][:topn] for row in pooled]


# ------------------------------------------------------------------ 评测
def rank_metrics(ranked: Sequence[Sequence[str]], golds: Sequence[str], *,
                 ks: Sequence[int] = (1, 5, 10, 20), ndcg_k: int = 5) -> Dict[str, float]:
    return RE.metrics_dict(ranked, golds, ks=ks, ndcg_k=ndcg_k)


def read_answers(reader, bundle: Bundle, ranked: Sequence[Sequence[str]],
                 order: str = "natural", max_chars: int = 2000,
                 gold_for_order: bool = True, progress: str = "") -> Dict[str, object]:
    """把排名拼成上下文 → 阅读器作答 → 与 ``answer_text`` 比 F1/EM。

    返回逐题记录（含上下文诊断），E5 靠 ``gold_used`` 区分"金标没进上下文"与"读错了"。
    """
    from reader.context_builder import build_context, get_last_stats

    pairs, rows = [], []
    for i, ranked_row in enumerate(ranked):
        gold = bundle.golds[i]
        ctx, used = build_context(
            list(ranked_row), bundle.doc_map, order=order, max_chars=max_chars,
            gold_doc_id=gold if gold_for_order else None,
        )
        stats = get_last_stats()
        pairs.append((bundle.questions[i], ctx))
        rows.append({"qid": bundle.qids[i], "gold": gold, "ranked": list(ranked_row),
                     "used": used, "ctx_chars": stats["chars"],
                     "gold_used": stats["gold_used"], "gold_position": stats["gold_position"],
                     "gold_in_candidates": stats["gold_in_candidates"]})

    preds = reader.answer_batch(pairs)
    f1s, ems = [], []
    for i, (row, (pred, score)) in enumerate(zip(rows, preds)):
        row.update({"pred": pred, "score": round(float(score), 4),
                    "f1": M.f1(pred, bundle.refs[i]), "em": M.em(pred, bundle.refs[i])})
        f1s.append(row["f1"])
        ems.append(row["em"])
    if progress:
        print(f"[reader] {progress} n={len(rows)} F1={sum(f1s) / len(f1s):.4f}", flush=True)
    return {"rows": rows, "f1": M.mean(f1s), "em": M.mean(ems)}


def summarize_reading(result: Dict[str, object]) -> Dict[str, float]:
    """把逐题记录压成一组数字（含"金标是否被保留"与条件 F1）。"""
    rows = result["rows"]
    kept = [r for r in rows if r["gold_used"]]
    pos = [r["gold_position"] for r in kept]
    return {
        "f1": round(result["f1"], 6),
        "em": round(result["em"], 6),
        "gold_kept_rate": round(len(kept) / max(1, len(rows)), 6),
        "gold_kept_f1": round(M.mean([r["f1"] for r in kept]), 6) if kept else 0.0,
        "gold_position_mean": round(M.mean(pos), 3) if pos else 0.0,
        "ctx_chars_mean": round(M.mean([r["ctx_chars"] for r in rows]), 1),
    }


def median(values: Sequence[float]) -> float:
    return round(statistics.median(values), 3) if values else 0.0


def bootstrap_f1(rows: Sequence[dict], n: int = 1000, alpha: float = 0.05,
                 seed: int = 42) -> List[float]:
    lo, hi = M.bootstrap_ci([r["f1"] for r in rows], n=n, alpha=alpha, seed=seed)
    return [round(lo, 6), round(hi, 6)]
