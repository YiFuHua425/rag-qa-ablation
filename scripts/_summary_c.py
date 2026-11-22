"""把 E4 / E5 的 summary 打成 markdown 表，方便贴进报告与交接单。

    python scripts/_summary_c.py e4
    python scripts/_summary_c.py e5
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"


def fmt(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, (int, float)):
        return f"{v:.{nd}f}"
    return str(v)


def e4():
    data = json.loads((RAW / "E4_summary.json").read_text(encoding="utf-8"))
    print("| run | depth | R@1 | R@5 | MRR | nDCG@5 | F1 | F1 95%CI | 金标保留 | 重排 ms | 总 ms |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in data["rows"]:
        ci = r.get("f1_ci") or [None, None]
        print(f"| {r['run_id']} | {r['depth'] if r['depth'] else 'off'} | "
              f"{fmt(r['recall@1'], 4)} | {fmt(r['recall@5'], 4)} | {fmt(r['mrr'], 4)} | "
              f"{fmt(r['ndcg@5'], 4)} | {fmt(r.get('f1'), 4)} | "
              f"{fmt(ci[0])}–{fmt(ci[1])} | {fmt(r.get('gold_kept_rate'))} | "
              f"{fmt(r.get('rerank_ms'), 0)} | {fmt(r.get('latency_ms'), 0)} |")
    print("\nconfig:", json.dumps(data.get("config"), ensure_ascii=False))


def e5():
    data = json.loads((RAW / "E5_summary.json").read_text(encoding="utf-8"))
    print("| run | order | max_chars | F1 | F1 95%CI | F1(保金标) | 金标保留 | 平均位次 | 上下文均长 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in data["rows"]:
        ci = r.get("f1_ci") or [None, None]
        print(f"| {r['run_id']} | {r['order']} | {r['max_chars']} | {fmt(r.get('f1'), 4)} | "
              f"{fmt(ci[0])}–{fmt(ci[1])} | {fmt(r.get('gold_kept_f1'), 4)} | "
              f"{fmt(r.get('gold_kept_rate'))} | {fmt(r.get('gold_position_mean'), 2)} | "
              f"{fmt(r.get('ctx_chars_mean'), 0)} |")
    print("\nconfig:", json.dumps(data.get("config"), ensure_ascii=False))


if __name__ == "__main__":
    (e4 if sys.argv[1:2] == ["e4"] else e5)()
