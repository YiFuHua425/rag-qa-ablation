#!/usr/bin/env python
"""CMRC2018 → corpus.jsonl / dev.jsonl（负责人：D）

用法::

    python data/build_data.py                    # 默认读 data/raw/cmrc2018_*.json
    python data/build_data.py --sample 20        # 额外导出 20 条人工核对样本
    python data/build_data.py --raw-dir data/raw --out-dir data

产物（格式见 INTERFACE.md 第 1 节）::

    data/corpus.jsonl   {"doc_id", "text"}                    全部唯一段落
    data/dev.jsonl      {"qid","question","gold_doc_id",
                         "answer_text","answer_start"}        dev 全部问题

为什么要自己写而不是直接用 datasets 库：
  1) squad-style 原始 json 的 answer_start 有少量偏移，本脚本用 ``context.find``
     校正并记账（报告第 4 章「数据清洗」直接用这个数）；
  2) 段落要跨 train/dev/trial **去重**后统一编号，doc_id 必须与金标一一对应。

数据闸门：跑完必须执行 ``python eval/check_data.py --strict`` 且四项全绿。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 语料库的取用顺序：先 train（最大），再 dev，再 trial。
# 去重时先到先得，因此同一段落永远得到同一个 doc_id。
CORPUS_SPLITS = ("train", "dev", "trial")
QUERY_SPLIT = "dev"          # 只有 dev 集带答案且不参与训练，作为评测集


def read_squad(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def iter_paragraphs(obj: dict):
    for article in obj.get("data", []):
        for para in article.get("paragraphs", []):
            yield para


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def build(raw_dir: Path, out_dir: Path, sample: int = 0) -> dict:
    stats = {
        "source_files": {},
        "paragraphs_raw": 0,
        "paragraphs_unique": 0,
        "paragraphs_dedup": 0,
        "questions_raw": 0,
        "questions_kept": 0,
        "questions_rejected": 0,
        "answer_start_corrected": 0,
        "answers_not_in_context": 0,
        "answers_empty": 0,
        "cross_split_overlap": {},
    }
    rejected: list[dict] = []

    # ---------- 1. 段落去重 + 统一编号 ----------
    doc_texts: list[str] = []
    index: dict[str, str] = {}
    for split in CORPUS_SPLITS:
        path = raw_dir / f"cmrc2018_{split}.json"
        if not path.exists():
            raise SystemExit(f"[X] 缺少原始文件 {path}（见 data/README.md 的下载说明）")
        obj = read_squad(path)
        n_before = len(doc_texts)
        n_para = 0
        for para in iter_paragraphs(obj):
            n_para += 1
            stats["paragraphs_raw"] += 1
            ctx = (para.get("context") or "").strip()
            if not ctx:
                continue
            if ctx in index:
                continue
            doc_id = f"p_{len(doc_texts) + 1:05d}"
            index[ctx] = doc_id
            doc_texts.append(ctx)
        stats["source_files"][split] = path.name
        stats["cross_split_overlap"][split] = {
            "paragraphs": n_para,
            "added_unique": len(doc_texts) - n_before,
        }

    stats["paragraphs_unique"] = len(doc_texts)
    stats["paragraphs_dedup"] = stats["paragraphs_raw"] - len(doc_texts)

    corpus_rows = [{"doc_id": index[t], "text": t} for t in doc_texts]
    index_inv = {row["doc_id"]: row["text"] for row in corpus_rows}

    # ---------- 2. dev 问题 → 金标段落 ----------
    dev_obj = read_squad(raw_dir / f"cmrc2018_{QUERY_SPLIT}.json")
    dev_rows: list[dict] = []
    for para in iter_paragraphs(dev_obj):
        ctx = (para.get("context") or "").strip()
        gold_doc_id = index.get(ctx)
        if gold_doc_id is None:                      # 理论上不会发生
            stats["questions_rejected"] += 1
            rejected.append({"reason": "gold paragraph missing", "question": para.get("qas")})
            continue
        for qa in para.get("qas", []):
            stats["questions_raw"] += 1
            question = (qa.get("question") or "").strip()
            answers = qa.get("answers") or []
            answer_text = (answers[0].get("text") or "").strip() if answers else ""
            src_start = answers[0].get("answer_start") if answers else None

            if not question:
                stats["questions_rejected"] += 1
                rejected.append({"reason": "empty question", "source_id": qa.get("id")})
                continue
            if not answer_text:
                stats["questions_rejected"] += 1
                stats["answers_empty"] += 1
                rejected.append({"reason": "empty answer", "source_id": qa.get("id")})
                continue

            pos = ctx.find(answer_text)              # 以原文为准，不信 answer_start
            if pos < 0:
                stats["questions_rejected"] += 1
                stats["answers_not_in_context"] += 1
                rejected.append({"reason": "answer not in gold paragraph",
                                 "source_id": qa.get("id"), "answer": answer_text})
                continue
            if src_start != pos:
                stats["answer_start_corrected"] += 1

            dev_rows.append({
                "qid": f"{QUERY_SPLIT}_{len(dev_rows) + 1:04d}",
                "question": question,
                "gold_doc_id": gold_doc_id,
                "answer_text": answer_text,
                "answer_start": pos,
                "source_id": qa.get("id"),           # 便于溯源，下游可忽略
            })
            stats["questions_kept"] += 1

    # ---------- 3. 落盘 ----------
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "corpus.jsonl", corpus_rows)
    write_jsonl(out_dir / "dev.jsonl", dev_rows)

    stats["out"] = {
        "corpus": str((out_dir / "corpus.jsonl").relative_to(ROOT)),
        "dev": str((out_dir / "dev.jsonl").relative_to(ROOT)),
    }
    stats["rejected"] = rejected[:50]
    with (out_dir / "raw" / "build_stats.json").open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)

    # ---------- 4. 人工核对样本（等距抽样，带金标段落上下文） ----------
    if sample > 0:
        step = max(1, len(dev_rows) // sample)
        picked = dev_rows[::step][:sample]
        lines = []
        for i, row in enumerate(picked, start=1):
            ctx = index_inv[row["gold_doc_id"]]
            s, e = row["answer_start"], row["answer_start"] + len(row["answer_text"])
            left = ctx[max(0, s - 40):s]
            right = ctx[e:e + 40]
            lines.append(
                f"[{i:02d}] qid={row['qid']}  gold={row['gold_doc_id']}\n"
                f"     Q: {row['question']}\n"
                f"     A: {row['answer_text']}  (start={row['answer_start']})\n"
                f"     段落: ...{left}【{row['answer_text']}】{right}...\n")
        (out_dir / "raw" / "manual_check_sample.txt").write_text(
            "\n".join(lines), encoding="utf-8")

    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="CMRC2018 清洗成 corpus.jsonl / dev.jsonl")
    ap.add_argument("--raw-dir", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--out-dir", default=str(ROOT / "data"))
    ap.add_argument("--sample", type=int, default=20, help="额外导出 N 条人工核对样本")
    args = ap.parse_args()

    stats = build(Path(args.raw_dir), Path(args.out_dir), args.sample)

    print("=" * 62)
    print("CMRC2018 → corpus.jsonl / dev.jsonl")
    print("=" * 62)
    print(f"  原始段落       {stats['paragraphs_raw']:>7d}")
    print(f"  去重后段落     {stats['paragraphs_unique']:>7d}   （去掉重复 {stats['paragraphs_dedup']}）")
    print(f"  原始问题       {stats['questions_raw']:>7d}")
    print(f"  保留问题       {stats['questions_kept']:>7d}")
    print(f"  丢弃问题       {stats['questions_rejected']:>7d}"
          f"   （空答案 {stats['answers_empty']}，答案不在段落 {stats['answers_not_in_context']}）")
    print(f"  answer_start 校正 {stats['answer_start_corrected']:>5d} 题（原始标注偏移）")
    print("-" * 62)
    print(f"  → {stats['out']['corpus']}")
    print(f"  → {stats['out']['dev']}")
    print("-" * 62)
    print("下一步：python eval/check_data.py --strict")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
