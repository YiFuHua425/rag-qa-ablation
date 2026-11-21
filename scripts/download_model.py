"""下载 BGE 编码器的 ONNX 权重（BGE-small-zh-v1.5，4 层 / 512 维）。

本仓库的检索链路走 onnxruntime（不依赖 torch），权重来自 Xenova 的 ONNX 导出：

    python scripts/download_model.py              # int8（22.8 MB），默认，够跑全部实验
    python scripts/download_model.py --fp32       # 另加 fp32（94.9 MB，量化对照用）
    python scripts/download_model.py --parts 4    # 大文件用 4 连接分段续传（推荐）

产物目录：``models/bge-small-zh-v1.5-onnx/``（已被 ``models/.gitignore`` 排除，不入库）。
"""
from __future__ import annotations

import argparse
import ssl
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = "Xenova/bge-small-zh-v1.5"
SMALL = [
    "config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
]
WEIGHTS = {"int8": "onnx/model_int8.onnx", "fp32": "onnx/model.onnx"}


def fetch_small(opener, url: str, local: Path, force: bool) -> None:
    if local.exists() and not force and local.stat().st_size > 0:
        print(f"[skip] {local.name} ({local.stat().st_size / 1e6:.2f} MB)")
        return
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with opener.open(req, timeout=300) as resp:
        data = resp.read()
    local.write_bytes(data)
    print(f"[ok  ] {local.name} ({local.stat().st_size / 1e6:.2f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", default="https://hf-mirror.com")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "models" / "bge-small-zh-v1.5-onnx"))
    ap.add_argument("--fp32", action="store_true", help="同时下载 fp32 权重（94.9 MB）")
    ap.add_argument("--parts", type=int, default=1, help=">1 时用 scripts/_dl.py 多连接分段下载权重")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    dst = Path(args.out)
    dst.mkdir(parents=True, exist_ok=True)
    base = f"{args.mirror.rstrip('/')}/{REPO}/resolve/main"
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))

    for name in SMALL:
        fetch_small(opener, f"{base}/{name}", dst / name, args.force)

    targets = ["int8"] + (["fp32"] if args.fp32 else [])
    dl_script = Path(__file__).resolve().parent / "_dl.py"
    for kind in targets:
        rel = WEIGHTS[kind]
        local = dst / Path(rel).name
        if local.exists() and local.stat().st_size > 1_000_000 and not args.force:
            print(f"[skip] {local.name} ({local.stat().st_size / 1e6:.1f} MB)")
            continue
        url = f"{base}/{rel}"
        print(f"[get ] {rel} → {local.name}")
        if args.parts > 1:
            subprocess.check_call([sys.executable, str(dl_script), url, str(local), "--parts", str(args.parts)])
        else:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with opener.open(req, timeout=600) as resp, open(local, "wb") as fh:
                while True:
                    buf = resp.read(1 << 20)
                    if not buf:
                        break
                    fh.write(buf)
            print(f"[ok  ] {local.name} ({local.stat().st_size / 1e6:.1f} MB)")

    print(f"完成：{dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
