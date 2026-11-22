"""fp32 vs int8 重排的一致性 + 速度对照（附录 C.3 用）。"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import c_pipeline as CP
from eval.io_utils import load_config
from rerank.cross_encoder import Reranker

cfg = load_config()
bundle = CP.build_candidates(cfg, split="smoke", alpha=0.2, depth=100, limit=10)
N = 20

def run(tag, path, backend):
    r = Reranker(device="cpu", threads=1, score_threads=18, model_path=path,
                 backend=backend, max_length=512).set_corpus(bundle.docs)
    r._ensure_model()
    t0 = time.time()
    tops = []
    for q, cand in zip(bundle.questions, bundle.candidates):
        out = r.score_candidates(q, cand[:N])
        tops.append([d for d, _ in out[:5]])
    dt = time.time() - t0
    return tops, dt

t_fp, s_fp = run("fp32", None, "hf")
t_i8, s_i8 = run("int8", "models/bge-reranker-base-onnx", "onnx")
overlap = sum(len(set(a) & set(b)) for a, b in zip(t_fp, t_i8))
print(f"fp32: {s_fp:.1f}s ({s_fp / 10 / N * 1000:.0f} ms/pair)  int8: {s_i8:.1f}s "
      f"({s_i8 / 10 / N * 1000:.0f} ms/pair)")
print(f"top-5 一致率 {overlap}/{len(t_fp) * 5} = {overlap / (len(t_fp) * 5):.3f}")
for i, (a, b) in enumerate(zip(t_fp, t_i8)):
    print(i, "same" if a == b else f"fp32={a} int8={b}")
