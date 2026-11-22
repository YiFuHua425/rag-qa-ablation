"""阅读器冒烟：只用金标段落做上下文，F1 应该很高（>0.8）。模型/解码有问题时这里先炸。"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.io_utils import load_config, load_split
from eval import metrics as M
from reader.context_builder import build_context
from reader.extractive_qa import ExtractiveReader

cfg = load_config()
docs, queries = load_split("smoke", cfg)
doc_map = {d["doc_id"]: d["text"] for d in docs}
reader = ExtractiveReader(device="cpu", batch_size=4, threads=18)
t0 = time.time()
pairs, refs = [], []
for q in queries[:5]:
    ctx, _ = build_context([q["gold_doc_id"]], doc_map, "natural", 2000, q["gold_doc_id"])
    pairs.append((q["question"], ctx))
    refs.append(q["answer_text"])
out = reader.answer_batch(pairs)
print(f"5 题耗时 {time.time() - t0:.1f}s，features/题={reader.last_stats}")
f1s = []
for (pred, score), ref, q in zip(out, refs, queries[:5]):
    f1s.append(M.f1(pred, ref))
    print(f"Q: {q['question']}\n  pred={pred!r} (score={score:.2f})  ref={ref!r}  f1={f1s[-1]:.2f}")
print("mean F1", round(sum(f1s) / len(f1s), 3))
