"""把"要推给 GitHub 的 B 相关文件"打成一个保持相对路径的 zip。

- 新增文件：B 自己的代码/脚本/文档/表/图
- 修改文件：4 个 [STUB] 实现 + eval/run_e1.py + eval/run_exp.py + requirements.txt + docs/pitfalls.md
- 不含被 .gitignore 挡住的东西（pptx / docx / results/raw / data/*.jsonl / models）
- 数据由第 1 棒 D 的 `data/build_data.py` 产出，B 不再自带造数脚本

用法：python scripts/make_push_bundle.py [输出目录]
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

ADDED = [
    "retrieval/tokenization.py",
    "retrieval/chunking.py",
    "retrieval/onnx_encoder.py",
    "retrieval/bert_tokenizer.py",
    "retrieval/hash_encoder.py",
    "eval/io_utils.py",
    "eval/retrieval_eval.py",
    "eval/encoders.py",
    "eval/plots.py",
    "eval/run_e2.py",
    "eval/run_e3.py",
    "eval/run_e6_rm3.py",
    "eval/run_extras_b.py",
    "eval/run_backend_check.py",
    "eval/run_quant_check.py",
    "tests/test_retrieval.py",
    "scripts/_dl.py",
    "scripts/download_model.py",
    "scripts/make_ppt_b.py",
    "scripts/inspect_ppt_b.py",
    "scripts/check_text_fit.py",
    "scripts/md_to_docx_b.py",
    "scripts/make_push_bundle.py",
    "models/.gitignore",
    "docs/B_检索模块说明.md",
    "docs/B_交接单.md",
    "docs/B_自查报告.md",
    "report/第2章_相关工作.md",
    "report/附录B_检索实验记录.md",
    "ppt/讲稿_B.md",
    "results/tab/E2_alpha.csv",
    "results/tab/E2_extras.csv",
    "results/tab/E2_backend.csv",
    "results/tab/E2_quant.csv",
    "results/tab/E3_chunk.csv",
    "results/tab/E6_rm3.csv",
    "results/fig/E2_alpha_recall.png",
    "results/fig/E2_latency.png",
    "results/fig/E3_chunk.png",
    "results/fig/E6_rm3.png",
]

MODIFIED = [
    "retrieval/bm25.py",
    "retrieval/bge_encoder.py",
    "retrieval/rrf.py",
    "retrieval/__init__.py",
    "eval/run_exp.py",
    "eval/run_e1.py",
    "requirements.txt",
    "docs/pitfalls.md",
]


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO.parent / "B_检索_交付" / "05_推送用_补丁包"
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = out_dir / "B_推仓库_文件包.zip"

    missing, added = [], 0
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in ADDED + MODIFIED:
            src = REPO / rel
            if not src.exists():
                missing.append(rel)
                continue
            zf.write(src, arcname=rel)
            added += 1

    size_mb = bundle.stat().st_size / 1e6
    print(f"[bundle] {bundle}  文件 {added} 个，{size_mb:.2f} MB")
    if missing:
        print("[bundle] 缺少：", *missing, sep="\n  ")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
