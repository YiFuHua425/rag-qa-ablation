#!/usr/bin/env python
"""数据闸门 —— 负责人：D（数据与评测）

用法::

    python eval/check_data.py                 # 体检，只看报告
    python eval/check_data.py --strict        # 有任何失败即 exit 1（提交前必跑）
    python eval/check_data.py --make-smoke 500 # 生成固定种子的 smoke 子集

为什么需要它：**答案不在金标段落里，后面所有指标都在算错东西。**
这道闸门不过，后面所有实验白跑。

四项断言（对应计划第 1 棒的"做到什么程度"）：
  1. 文件存在、能按 jsonl 解析、字段齐全
  2. doc_id / qid 唯一；text 非空
  3. 每题 gold_doc_id 在 corpus 中存在 —— 缺失数必须为 0
  4. answer_text 是其金标段落的子串 —— 越界数必须为 0
     （并校验 answer_start 指向的位置确实是答案）
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"[X] {path.name} 第 {lineno} 行不是合法 JSON：{exc}")
    return rows


def check(corpus_path: Path, dev_path: Path) -> tuple[bool, dict]:
    problems: list[str] = []
    stats: dict = {}

    if not corpus_path.exists():
        problems.append(f"缺少 {corpus_path.relative_to(ROOT)}")
    if not dev_path.exists():
        problems.append(f"缺少 {dev_path.relative_to(ROOT)}")
    if problems:
        return False, {"problems": problems}

    corpus = load_jsonl(corpus_path)
    dev = load_jsonl(dev_path)

    # --- 1. 字段齐全 ---
    for i, d in enumerate(corpus):
        for key in ("doc_id", "text"):
            if key not in d:
                problems.append(f"corpus 第 {i+1} 条缺字段 {key}")
                break
    for i, q in enumerate(dev):
        for key in ("qid", "question", "gold_doc_id", "answer_text", "answer_start"):
            if key not in q:
                problems.append(f"dev 第 {i+1} 条缺字段 {key}")
                break

    doc_map = {d.get("doc_id"): d.get("text", "") for d in corpus}

    # --- 2. 唯一性与非空 ---
    dup_doc = [k for k, v in Counter(d.get("doc_id") for d in corpus).items() if v > 1]
    dup_q = [k for k, v in Counter(q.get("qid") for q in dev).items() if v > 1]
    if dup_doc:
        problems.append(f"corpus doc_id 重复 {len(dup_doc)} 个，例：{dup_doc[:3]}")
    if dup_q:
        problems.append(f"dev qid 重复 {len(dup_q)} 个，例：{dup_q[:3]}")
    empty_text = sum(1 for d in corpus if not (d.get("text") or "").strip())
    if empty_text:
        problems.append(f"corpus 有 {empty_text} 条 text 为空")

    # --- 3. gold_doc_id 必须在 corpus 中 ---
    missing_gold = [q["qid"] for q in dev if q.get("gold_doc_id") not in doc_map]
    if missing_gold:
        problems.append(
            f"gold_doc_id 在 corpus 中缺失 {len(missing_gold)} 题，例：{missing_gold[:3]}")

    # --- 4. answer_text 必须是金标段落子串（最关键） ---
    out_of_span = []
    start_mismatch = []
    for q in dev:
        gold_text = doc_map.get(q.get("gold_doc_id"), "")
        ans = q.get("answer_text") or ""
        if not ans:
            out_of_span.append(q.get("qid"))
            continue
        if ans not in gold_text:
            out_of_span.append(q.get("qid"))
            continue
        start = q.get("answer_start")
        if isinstance(start, int) and 0 <= start < len(gold_text):
            if gold_text[start:start + len(ans)] != ans:
                start_mismatch.append(q.get("qid"))
        else:
            start_mismatch.append(q.get("qid"))
    if out_of_span:
        problems.append(
            f"[致命] answer_text 不在金标段落中 {len(out_of_span)} 题，例：{out_of_span[:3]}")
    if start_mismatch:
        problems.append(
            f"answer_start 与答案位置不符 {len(start_mismatch)} 题，例：{start_mismatch[:3]}")

    lens = [len(t) for t in doc_map.values() if t]
    ans_lens = [len(q.get("answer_text") or "") for q in dev]

    stats.update({
        "corpus_file": str(corpus_path.relative_to(ROOT)),
        "dev_file": str(dev_path.relative_to(ROOT)),
        "n_docs": len(corpus),
        "n_questions": len(dev),
        "paragraph_len_mean": round(sum(lens) / len(lens), 1) if lens else 0,
        "paragraph_len_max": max(lens) if lens else 0,
        "answer_len_mean": round(sum(ans_lens) / len(ans_lens), 1) if ans_lens else 0,
        "n_unique_gold": len({q.get("gold_doc_id") for q in dev}),
        "answer_out_of_span": len(out_of_span),
        "answer_start_mismatch": len(start_mismatch),
        "problems": problems,
    })
    return (not problems), stats


def make_smoke(dev_path: Path, n: int, seed: int = 42) -> Path:
    dev = load_jsonl(dev_path)
    rng = random.Random(seed)
    picked = rng.sample(dev, min(n, len(dev)))
    out = DATA / "dev_smoke.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for row in picked:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="数据闸门")
    ap.add_argument("--corpus", default=str(DATA / "corpus.jsonl"))
    ap.add_argument("--dev", default=str(DATA / "dev.jsonl"))
    ap.add_argument("--strict", action="store_true", help="有失败即 exit 1")
    ap.add_argument("--make-smoke", type=int, metavar="N", help="生成 N 题的 smoke 子集")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    corpus_path, dev_path = Path(args.corpus), Path(args.dev)

    if args.make_smoke:
        if not dev_path.exists():
            print(f"[X] 先有 {dev_path.name} 才能切 smoke 子集", file=sys.stderr)
            return 2
        out = make_smoke(dev_path, args.make_smoke, args.seed)
        print(f"[OK] 已生成 {out.relative_to(ROOT)}（{args.make_smoke} 题，seed={args.seed}）")

    ok, stats = check(corpus_path, dev_path)
    print("=" * 62)
    print("数据体检")
    print("=" * 62)
    if "problems" in stats and stats["problems"] and "n_docs" not in stats:
        for p in stats["problems"]:
            print(f"  [X] {p}")
        print("-" * 62)
        print("[WAIT] 数据未就绪 —— 这是第 1 棒（D）的活。")
        print("       下载 CMRC2018 → 清洗成 corpus.jsonl / dev.jsonl")
        print("       格式见 INTERFACE.md 第 1 节")
        return 1 if args.strict else 0

    for key in ("n_docs", "n_questions", "n_unique_gold", "paragraph_len_mean",
                "paragraph_len_max", "answer_len_mean"):
        print(f"  {key:22s} {stats[key]}")
    print("-" * 62)
    print(f"  answer 越界数           {stats['answer_out_of_span']}   (必须 = 0)")
    print(f"  answer_start 不符       {stats['answer_start_mismatch']}   (必须 = 0)")
    print("-" * 62)

    if ok:
        print("[OK] 四项断言全部通过，数据可以用于实验。")
        print("     下一步：python eval/check_data.py --make-smoke 500")
        return 0

    for p in stats["problems"]:
        print(f"  [X] {p}")
    print("-" * 62)
    print("[FAIL] 数据不合格。**不要开始跑实验**，先修数据。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())