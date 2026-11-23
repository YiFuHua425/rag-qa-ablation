#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""单题链路演示：一条命令看清「检索 → 融合 → 拼装 → 抽取」每一步。

设计目标（对应报告第 3 章的论证结构）：
    同一个问题，分别看 BM25 排第几、稠密排第几、融合后排第几、重排后又排第几，
    最后看阅读器把哪一段的哪几个字抽出来当答案。
    —— 一条命令讲完，就是整篇的消融逻辑。

用法：
    python demo.py                          # 自动挑一题「能说明问题」的（一路没召回/融合有提升）
    python demo.py --index 3                # 指定第 4 题
    python demo.py --qid dev_2620           # 指定题号（--list 可查）
    python demo.py --question "苹果公司的创始人是谁"   # 自定义问题（无金标，不算 F1）
    python demo.py --list                   # 列出可演示的题目
    python demo.py --alpha 0.2 --topk 3     # 改融合权重与展示条数
    python demo.py --rerank                 # 加一步交叉编码器重排（需 torch + 权重）
    python demo.py --no-reader              # 跳过阅读器（只演示检索侧）

依赖：
    检索侧只需 numpy + onnxruntime + jieba（无需 torch）；
    阅读器/重排需要 torch + transformers + 权重，缺失时自动跳过并说明原因。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:          # 允许 `python demo.py` 从仓库根直接跑
    sys.path.insert(0, str(ROOT))

# 管道/重定向时 Windows 默认用 GBK 编码，中文会乱码；统一按 UTF-8 输出
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

BAR = "═" * 80
SUB = "─" * 80


def _w(s, n: int) -> str:
    """按终端显示宽度左侧补空格（中日韩字符算 2 列），保证表格对齐。"""
    s = str(s)
    width = sum(2 if ord(c) > 0x2E80 else 1 for c in s)
    return s + " " * max(0, n - width)


def _cut(s, n: int) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def section(no: str, title: str, note: str = "") -> None:
    print(f"\n【{no}】{title}" + (f"　　{note}" if note else ""))


def rank_of(ids, target) -> int | None:
    return ids.index(target) + 1 if (target and target in ids) else None


def pick_interesting(queries, bm25, index, encoder, alpha, fuse_k, depth,
                     scan: int = 60, short_answer: int = 18):
    """自动挑一题「能说明问题」的，返回 (下标, 理由)。

    两遍扫描：
      ① 只在**短答案**（≤18 字）的题里找 —— 抽取式阅读器对长答案 F1 天生偏低，
         演示要看"答对了"才算成功；
      ② 找不到就放宽到全部题。
    优先级：一路没召回→另一路召回（融合能救）> 融合比单路都好 > 两路分歧大。
    """
    from retrieval.rrf import alpha_weights, fuse

    pool = [q for q in queries[:scan] if q.get("gold_doc_id")]
    if not pool:
        return 0, "（没有带金标的题）"
    embs = encoder.encode_queries([q["question"] for q in pool], show_progress=False)
    weights = alpha_weights(alpha)

    def _scan(cands):
        best = None
        for q in cands:
            idx = pool.index(q)
            gold = q["gold_doc_id"]
            s_ids = [d for d, _ in bm25.search(q["question"], topk=depth)]
            d_ids = [d for d, _ in index.search(embs[idx], topk=depth)]
            f_ids = [d for d, _ in fuse([[(x, 0.0) for x in s_ids],
                                         [(x, 0.0) for x in d_ids]],
                                        weights=weights, k=fuse_k)]
            rs, rd, rf = rank_of(s_ids, gold), rank_of(d_ids, gold), rank_of(f_ids, gold)
            fused_better = rf is not None and (rs is None or rf < rs) and (rd is None or rf < rd)
            one_missed = (rs is None) != (rd is None)
            if fused_better and one_missed:
                case = (0, f"一路没召回，融合把金标顶到第 {rf} 位（互补生效）")
            elif fused_better:
                case = (1, f"单路各第 {rs}/{rd} 位，融合后升到第 {rf} 位")
            elif one_missed:
                case = (2, f"一路没召回（{'BM25' if rs is None else '稠密'}），"
                           f"融合后金标第 {rf or '未召回'} 位")
            elif rs and rd and abs(rs - rd) >= 3:
                case = (3, f"两路分歧明显（BM25 第 {rs} / 稠密 第 {rd}）")
            else:
                continue
            key = (case[0], len(str(q.get("answer_text") or "")))
            if best is None or key < best[0]:
                best = (key, idx, case[1])
        return best

    short = [q for q in pool if len(str(q.get("answer_text") or "")) <= short_answer]
    best = _scan(short) if short else None
    tag = ""
    if best is None:
        best, tag = _scan(pool), "（已放宽答案长度）"
    if best is None:
        return 0, f"（前 {scan} 题没有明显分歧，用第 0 题）"
    return best[1], best[2] + tag


def main() -> int:
    ap = argparse.ArgumentParser(description="单题链路演示：检索 → 融合 → 拼装 → 抽取")
    ap.add_argument("--split", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--index", type=int, default=None, help="用第几题（0 起）")
    ap.add_argument("--qid", default=None, help="指定题号，优先级高于 --index")
    ap.add_argument("--question", default=None, help="自定义问题（无金标）")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--depth", type=int, default=20, help="两路召回深度（默认 20，够看排名）")
    ap.add_argument("--alpha", type=float, default=None, help="稠密路权重，默认取 config.yaml")
    ap.add_argument("--max-chars", type=int, default=None, help="上下文上限，默认取 config.yaml")
    ap.add_argument("--model-path", default=None, help="编码器权重（默认自动找 models/ 下的 int8）")
    ap.add_argument("--rerank", action="store_true", help="额外演示交叉编码器重排（需 torch）")
    ap.add_argument("--score-threads", type=int, default=None,
                    help="算分线程数（默认 cpu_count-2，与 run_e4 一致；延迟仍按单线程口径报）")
    ap.add_argument("--no-reader", action="store_true", help="跳过阅读器")
    ap.add_argument("--list", action="store_true", help="列出可演示的题目")
    args = ap.parse_args()

    t_all = time.perf_counter()
    # jieba 首次加载会往 stderr 打 "Building prefix dict..."，录屏时很难看，关掉
    try:
        import logging

        import jieba
        jieba.setLogLevel(logging.ERROR)
    except Exception:
        pass
    # transformers v5 加载权重时默认刷 "Loading weights: 100%|..." 进度条，一并静音
    try:
        import transformers
        transformers.logging.disable_progress_bar()
        transformers.logging.set_verbosity_error()
    except Exception:
        pass

    from eval.io_utils import load_config, load_split
    from retrieval.bge_encoder import BgeEncoder, FaissIndex
    from retrieval.bm25 import BM25
    from retrieval.rrf import alpha_weights, fuse
    from retrieval.tokenization import build_tokenizer

    cfg = load_config()
    docs, queries = load_split(args.split, cfg)
    doc_ids = [d["doc_id"] for d in docs]
    doc_texts = [d["text"] for d in docs]
    doc_map = dict(zip(doc_ids, doc_texts))

    if args.list:
        print(f"共 {len(queries)} 题（split={args.split}），前 15 题：")
        print(f"  {'#':>3}  {_w('qid', 16)}{_w('金标段落', 12)}问题")
        for i, q in enumerate(queries[:15]):
            print(f"  {i:>3}  {_w(q['qid'], 16)}{_w(q.get('gold_doc_id') or '-', 12)}"
                  f"{_cut(q['question'], 42)}")
        print(f"\n用法：python demo.py --index 3   或   python demo.py --qid <qid>")
        return 0

    fusion_cfg = cfg.get("retrieval", {}).get("fusion", {})
    alpha = float(args.alpha if args.alpha is not None else fusion_cfg.get("alpha", 0.5))
    fuse_k = int(fusion_cfg.get("k", 60))
    max_chars = int(args.max_chars if args.max_chars is not None
                    else cfg.get("context", {}).get("max_chars", 2000))
    topk = int(args.topk)
    depth = max(int(args.depth), topk)
    # 算分线程：与 run_e4 同一公式（cpu_count-2）。延迟仍按单线程口径单独测，
    # 否则在 24 核机器上重排会退化成 1 线程、慢 5 倍。
    import os as _os
    score_threads = int(args.score_threads) if args.score_threads else max(1, (_os.cpu_count() or 3) - 2)
    lat_cfg = cfg.get("latency", {}) or {}

    print(BAR)
    print(" 检索增强问答 · 单题链路演示")
    print(" BM25（稀疏）+ BGE（稠密）→ RRF 融合 → 交叉编码器重排 → 抽取式阅读器")
    print(BAR)

    # ---------------------------------------------------- 【0】装配
    section("0", "装配：语料 → 编码器 → 向量索引 → BM25 倒排")
    model_path = args.model_path
    if model_path is None:
        cand = ROOT / "models" / "bge-small-zh-v1.5-onnx" / "model_int8.onnx"
        model_path = str(cand) if cand.exists() else None
    dense_cfg = cfg.get("retrieval", {}).get("dense", {})
    encoder = BgeEncoder(
        model_name=dense_cfg.get("model", "BAAI/bge-small-zh-v1.5"),
        query_prefix=dense_cfg.get("query_prefix", ""),
        normalize=bool(dense_cfg.get("normalize", True)),
        batch_size=int(dense_cfg.get("encode_batch", 64)),
        model_path=model_path, backend="auto",
    )
    avg_len = sum(len(t) for t in doc_texts) // max(1, len(doc_texts))
    print(f"  语料 {len(docs)} 段（平均 {avg_len} 字）　维度 {encoder.dim} 维"
          f"　相似度 = L2 归一化后的内积（等价余弦）")

    t0 = time.perf_counter()
    index = FaissIndex().build_from_encoder(
        encoder, [{"doc_id": d, "text": t} for d, t in zip(doc_ids, doc_texts)],
        tag="demo_paragraph", show_progress=True,
    )
    t_idx = (time.perf_counter() - t0) * 1000
    print(f"  向量索引 {index.size} 篇 × {encoder.dim} 维　后端 {index.backend}"
          f"　耗时 {t_idx:.0f} ms（命中缓存时只是读盘）")

    sparse_cfg = cfg.get("retrieval", {}).get("sparse", {})
    tok_name = sparse_cfg.get("tokenizer", "jieba")
    tokenizer = build_tokenizer(None if tok_name in ("auto", "jieba") else tok_name)
    k1, b = float(sparse_cfg.get("k1", 1.5)), float(sparse_cfg.get("b", 0.75))
    bm25 = BM25(k1=k1, b=b, tokenizer=tokenizer)
    bm25.fit([{"doc_id": d, "text": t} for d, t in zip(doc_ids, doc_texts)])
    print(f"  BM25 倒排就绪　分词器 {tok_name}　k1={k1}　b={b}")

    # ---------------------------------------------------- 选题
    if args.question:
        query = {"qid": "(自定义)", "question": args.question,
                 "gold_doc_id": None, "answer_text": None}
        why = "（自定义问题，无金标）"
    elif args.qid:
        query = next((q for q in queries if q["qid"] == args.qid), None)
        if query is None:
            print(f"\n[X] 找不到 qid={args.qid}，用 --list 查看", file=sys.stderr)
            return 2
        why = "（指定题号）"
    elif args.index is not None:
        if not (0 <= args.index < len(queries)):
            print(f"\n[X] --index 超出范围（共 {len(queries)} 题）", file=sys.stderr)
            return 2
        query = queries[args.index]
        why = "（指定下标）"
    else:
        idx, why = pick_interesting(queries, bm25, index, encoder, alpha, fuse_k, depth)
        query = queries[idx]
        args.index = idx
        why = f"（自动挑选：{why}）"

    gold = query.get("gold_doc_id")
    gold_answer = query.get("answer_text")

    print(f"\n{SUB}")
    print(f" 问题：{query['question']}")
    print(f" 题号：{query['qid']}　split={args.split}　金标段落：{gold or '（无）'}")
    print(f" 选题：{why.strip('（）')}")
    print(SUB)

    # ---------------------------------------------------- 【1】分词
    section("1", "查询分词（BM25 用这个切词）")
    terms = bm25.query_terms(query["question"])
    print(f"  词项 {len(terms)} 个：{' / '.join(terms) if terms else '(空)'}")

    # ---------------------------------------------------- 【2】稀疏路
    section("2", "稀疏路：BM25", f"召回深度 {depth}，展示前 {topk}")
    sparse_hits = bm25.search(query["question"], topk=depth)
    sparse_ids = [d for d, _ in sparse_hits]
    print(f"  {_w('名次', 6)}{_w('doc_id', 12)}{'BM25 分':>10}   金标")
    for i, (did, sc) in enumerate(sparse_hits[:topk], 1):
        print(f"  {_w(i, 6)}{_w(did, 12)}{sc:>10.4f}   {'★' if did == gold else ''}")
    r_s = rank_of(sparse_ids, gold)
    print(f"  → 金标名次：{r_s or f'未进前 {depth}'}")

    # ---------------------------------------------------- 【3】稠密路
    section("3", "稠密路：BGE 编码 + 内积检索", f"召回深度 {depth}，展示前 {topk}")
    t0 = time.perf_counter()
    q_emb = encoder.encode_query(query["question"])
    dense_hits = index.search(q_emb, topk=depth)
    t_q = (time.perf_counter() - t0) * 1000
    dense_ids = [d for d, _ in dense_hits]
    print(f"  单查询（编码 + 检索）{t_q:.1f} ms")
    print(f"  {_w('名次', 6)}{_w('doc_id', 12)}{'余弦':>10}   金标")
    for i, (did, sc) in enumerate(dense_hits[:topk], 1):
        print(f"  {_w(i, 6)}{_w(did, 12)}{sc:>10.4f}   {'★' if did == gold else ''}")
    r_d = rank_of(dense_ids, gold)
    print(f"  → 金标名次：{r_d or f'未进前 {depth}'}")

    # ---------------------------------------------------- 【4】RRF
    section("4", "融合：RRF（只用排名，不用分数）",
            f"k={fuse_k}　α={alpha} → 权重 [稀疏 {1-alpha:.1f}, 稠密 {alpha:.1f}]")
    print("  score(d) = Σ wᵢ/(k + rankᵢ)；某一路没召回时该项记 0")
    fused = fuse([[(d, 0.0) for d in sparse_ids], [(d, 0.0) for d in dense_ids]],
                 weights=alpha_weights(alpha), k=fuse_k)
    fused_ids = [d for d, _ in fused]
    print(f"  {_w('名次', 6)}{_w('doc_id', 12)}{'BM25 名次':>10}{'稠密名次':>10}{'融合分':>12}   金标")
    for i, (did, sc) in enumerate(fused[:topk], 1):
        rs, rd = rank_of(sparse_ids, did), rank_of(dense_ids, did)
        print(f"  {_w(i, 6)}{_w(did, 12)}{str(rs or '-'):>10}{str(rd or '-'):>10}"
              f"{sc:>12.6f}   {'★' if did == gold else ''}")
    r_f = rank_of(fused_ids, gold)
    print(f"  → 金标名次：{r_f or '未召回'}")

    # ---------------------------------------------------- 【5】重排（可选）
    reranked_ids = None
    if args.rerank:
        section("5", "重排：交叉编码器（bge-reranker-base）", f"只重排前 {depth} 个候选")
        try:
            from rerank.cross_encoder import Reranker
            lat = cfg.get("latency", {}) or {}
            rcfg = cfg.get("rerank", {}) or {}
            rk = Reranker(model_name=rcfg.get("model", "BAAI/bge-reranker-base"),
                          depth=depth, topn=topk, device=str(lat_cfg.get("device", "cpu")),
                          max_length=int(rcfg.get("max_length", 512)),
                          threads=int(lat_cfg.get("threads", 1)),
                          score_threads=score_threads).set_doc_map(doc_map)
            t0 = time.perf_counter()
            out = rk.rerank(query["question"], fused[:depth], topn=topk)
            ms = (time.perf_counter() - t0) * 1000
            reranked_ids = [d for d, _ in out]
            print(f"  {_w('名次', 6)}{_w('doc_id', 12)}{'重排分':>10}   金标")
            for i, (did, sc) in enumerate(out, 1):
                print(f"  {_w(i, 6)}{_w(did, 12)}{sc:>10.4f}   {'★' if did == gold else ''}")
            n = max(1, min(depth, len(fused)))
            print(f"  耗时 {ms:.0f} ms（{score_threads} 线程算分，{ms/n:.0f} ms/候选）")
            print(f"  注：报告里的延迟口径是 CPU 单线程 ≈225 ms/候选（见 E4），此处为吞吐口径")
        except Exception as exc:
            first = (str(exc).strip().splitlines() or [type(exc).__name__])[0][:70]
            print(f"  [跳过] 重排不可用：{first}")
            print("         需要 torch + transformers + 权重"
                  "（python scripts/download_models_c.py --source modelscope --only rerank）")

    # ---------------------------------------------------- 【6】上下文
    final_ids = reranked_ids or fused_ids
    section("6", "上下文拼装（决定「检索对了能不能答对」）",
            f"max_chars={max_chars}　order=natural")
    from reader.context_builder import build_context, get_last_stats
    context, used = build_context(final_ids[:topk], doc_map, order="natural",
                                  max_chars=max_chars, gold_doc_id=gold)
    st = get_last_stats()
    print(f"  候选 {st['n_candidates']} 段 → 进入上下文 {st['n_used']} 段 / {st['chars']} 字"
          f"（上限 {st['max_chars']}）")
    if st["dropped_doc_ids"]:
        print(f"  被长度预算挡在门外：{', '.join(st['dropped_doc_ids'])}")
    if gold:
        if st["gold_used"]:
            print(f"  金标是否进入上下文：是，位置第 {st['gold_position']} 段")
        else:
            print(f"  金标是否进入上下文：否（没召回，或被截断）← 此时 F1 必然为 0")
    for i, did in enumerate(used, 1):
        print(f"    [{i}] {_w(did, 12)}{_cut(doc_map[did], 60)}")

    # ---------------------------------------------------- 【7】阅读器
    section("7", "抽取式阅读器（在上下文里定位答案片段）")
    pred = None
    if args.no_reader:
        print("  [跳过] --no-reader")
    else:
        try:
            from reader.extractive_qa import ExtractiveReader
            from eval.metrics import em as em_fn, f1 as f1_fn
            rcfg = cfg.get("reader", {}) or {}
            # 演示固定走 CPU：与第 5 章的延迟口径一致，也避免无 GPU 机器上报 CUDA 错
            dev = str(lat_cfg.get("device", "cpu"))
            reader = ExtractiveReader(
                model_name=rcfg.get("model", "uer/roberta-base-chinese-extractive-qa"),
                max_length=int(rcfg.get("max_length", 512)),
                device=dev,
                threads=int(lat_cfg.get("threads", 1)),
                score_threads=score_threads)
            t0 = time.perf_counter()
            pred, score = reader.answer(query["question"], context)
            ms = (time.perf_counter() - t0) * 1000
            print(f"  预测答案 {pred!r}　score {score:.4f}　{ms:.0f} ms")
            if gold_answer:
                print(f"  标准答案 {gold_answer!r}")
                print(f"  F1 {f1_fn(pred, gold_answer):.4f}　EM {em_fn(pred, gold_answer):.0f}")
        except Exception as exc:
            first = (str(exc).strip().splitlines() or [type(exc).__name__])[0][:70]
            print(f"  [跳过] 阅读器不可用：{first}")
            print("         需要 torch + transformers + 权重 407 MB："
                  "python scripts/download_models_c.py --source hf --only reader --parts 16")

    # ---------------------------------------------------- 【8】小结
    section("8", "小结（这张表就是全篇的论证结构）")
    if gold:
        final_name = "重排后" if reranked_ids else "RRF 融合"
        rows = [("BM25 单路", r_s), ("稠密单路", r_d), ("RRF 融合", r_f)]
        if reranked_ids:
            rows.append(("重排后", rank_of(reranked_ids, gold)))
        r_final = rank_of(reranked_ids or fused_ids, gold)
        print(f"  {_w('配置', 14)}{'金标名次':>10}")
        print(f"  {'-'*24}")
        for name, r in rows:
            print(f"  {_w(name, 14)}{(str(r) if r else '未召回'):>10}")

        singles = [r for r in (r_s, r_d) if r]
        best_single = min(singles) if singles else None
        # 与「上一步」比：有重排就跟融合比，否则跟最好的单路比
        if reranked_ids:
            prev_val, prev_name = r_f, "RRF 融合"
        else:
            prev_val, prev_name = best_single, "最好的单路"
        print(f"\n  → 实际用于回答的是「{final_name}」，金标名次 "
              f"{r_final if r_final else '未召回'}。")
        if r_final is None:
            print("     金标没进最终排名：纯检索失败，下游无从补救。")
        elif r_final == 1:
            if reranked_ids and r_f and r_f > 1:
                print(f"     重排把金标从融合后的第 {r_f} 位顶到第 1 位 —— 精排在这一题上决定性地救回来了。")
            elif not reranked_ids and prev_val and prev_val > 1:
                print(f"     {prev_name} 把金标从第 {prev_val} 位提升到第 1 位 —— 混合检索的收益。")
            else:
                print("     金标已在第 1 位，检索侧无损失——答对与否只取决于阅读器与上下文预算。")
        elif prev_val and r_final < prev_val:
            print(f"     {prev_name} 把金标放在第 {prev_val} 位，{final_name} 提升到第 {r_final} 位"
                  f"—— 这一步是有效收益。")
        elif prev_val and r_final > prev_val:
            print(f"     ⚠ {final_name} 反而把金标从第 {prev_val} 位挤到第 {r_final} 位。")
            if reranked_ids:
                print("       单题上精排可能变差（交叉编码器只建模局部相关性）；")
                print("       E4 报的 R@1 +9 个点是 100 题平均，这一题正好用来说明「精排不是万灵药」。")
                print("       想看重排有效的样本：换题再跑，如 python demo.py --rerank --index 2")
            else:
                print(f"       说明 α={alpha} 对本语料不是最优（E2 扫出 α=0.2 最好），")
                print("       可加 --alpha 0.2 复现报告的最优配置后再看。")
        else:
            print(f"     {final_name} 与 {prev_name} 名次相同：这一步没有改变排序。")
    else:
        print("  自定义问题没有金标：跳过名次对比，上面各步的 top-k 即最终结果。")

    print(f"\n{SUB}")
    print(f" 全程 {time.perf_counter() - t_all:.1f} s")
    print(SUB)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())