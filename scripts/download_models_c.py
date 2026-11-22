"""下载 C 需要的两个权重（重排 + 抽取式阅读器）—— 负责人：C

    python scripts/download_models_c.py --source modelscope --only rerank
    python scripts/download_models_c.py --source hf --only reader --parts 12

产物（``models/.gitignore`` 已排除，不入库）::

    models/bge-reranker-base/                     config + tokenizer + model.safetensors (1.1 GB)
    models/roberta-base-chinese-extractive-qa/    config + vocab + pytorch_model.bin (0.39 GB)

镜像说明（本机实测，2026-11）：

===============  ==================================  ====================================
源               URL 格式                             实测
===============  ==================================  ====================================
modelscope       ``/models/<repo>/resolve/master/``   单连接 7–12 MB/s，**但不支持 HEAD/Range**
hf-mirror.net    ``/<repo>/resolve/main/``            **直出文件不走 xethub，1–4 MB/s**（首选）
hf-mirror.com    ``/<repo>/resolve/main/``            302 到 cas-bridge.xethub.hf.co，
                                                     该域常被墙：20–300 KB/s 甚至 0
===============  ==================================  ====================================

所以：bge-reranker-base（ModelScope 有镜像）走 modelscope；
uer 的阅读器（ModelScope 没有）走 **hf-mirror.net**——`.com` 会把请求 302 到 xethub 的
CAS 节点，那个域名在本机几乎不通（这也是第一次跑 E4 时 407 MB 下不下来的真正原因，
不是"网络慢"，是"走错了节点"）。
不用 ``scripts/_dl.py`` 的地方只有一处差别：modelscope 的 HEAD 返回 ``Content-Length: 0``，
分段逻辑没法用，直接单流下载。
"""
from __future__ import annotations

import argparse
import ssl
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DL = Path(__file__).resolve().parent / "_dl.py"

SMALL = ["config.json", "tokenizer_config.json", "special_tokens_map.json"]
MODELS = {
    "rerank": {
        "repo": "BAAI/bge-reranker-base",
        "out": ROOT / "models" / "bge-reranker-base",
        "small": SMALL + ["tokenizer.json", "sentencepiece.bpe.model"],
        "weights": ["model.safetensors"],
    },
    "reader": {
        "repo": "uer/roberta-base-chinese-extractive-qa",
        "out": ROOT / "models" / "roberta-base-chinese-extractive-qa",
        "small": SMALL + ["vocab.txt"],
        "weights": ["pytorch_model.bin"],
    },
}
UA = {"User-Agent": "Mozilla/5.0"}


def _opener():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))


def fetch_small(opener, url: str, local: Path, force: bool) -> None:
    if local.exists() and local.stat().st_size > 0 and not force:
        print(f"[skip] {local.name} ({local.stat().st_size / 1e6:.2f} MB)")
        return
    with opener.open(urllib.request.Request(url, headers=UA), timeout=300) as resp:
        data = resp.read()
    local.write_bytes(data)
    print(f"[ok  ] {local.name} ({len(data) / 1e6:.2f} MB)")


def fetch_stream(opener, url: str, local: Path) -> None:
    """单流下载 + 每秒一行进度（modelscope 用这条路）。"""
    t0 = time.time()
    done = 0
    with opener.open(urllib.request.Request(url, headers=UA), timeout=900) as resp, \
            open(local, "wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        while True:
            buf = resp.read(1 << 20)
            if not buf:
                break
            fh.write(buf)
            done += len(buf)
            if total:
                print(f"\r[dl] {done / 1e6:7.1f}/{total / 1e6:.1f} MB "
                      f"{done / 1e6 / max(1e-9, time.time() - t0):6.2f} MB/s", end="", flush=True)
    print(f"\n[ok  ] {local.name} ({local.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser(description="下载 bge-reranker-base 与中文抽取式 QA 权重")
    ap.add_argument("--source", default="hf", choices=["hf", "modelscope"])
    ap.add_argument("--only", default="all", choices=["all", "rerank", "reader"])
    ap.add_argument("--parts", type=int, default=4, help="hf 源的大文件分段数（modelscope 忽略）")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    opener = _opener()
    keys = ["rerank", "reader"] if args.only == "all" else [args.only]
    for key in keys:
        spec = MODELS[key]
        dst: Path = spec["out"]
        dst.mkdir(parents=True, exist_ok=True)
        if args.source == "modelscope":
            base = f"https://modelscope.cn/models/{spec['repo']}/resolve/master"
        else:
            base = f"https://hf-mirror.net/{spec['repo']}/resolve/main"
        print(f"\n== {key}: {spec['repo']} → {dst.relative_to(ROOT)}（{args.source}）")
        for name in spec["small"]:
            fetch_small(opener, f"{base}/{name}", dst / name, args.force)
        for name in spec["weights"]:
            local = dst / name
            if local.exists() and local.stat().st_size > 10_000_000 and not args.force:
                print(f"[skip] {name} ({local.stat().st_size / 1e6:.1f} MB)")
                continue
            print(f"[get ] {name}")
            if args.source == "modelscope" or args.parts <= 1:
                fetch_stream(opener, f"{base}/{name}", local)
            else:
                subprocess.check_call([sys.executable, str(DL), f"{base}/{name}",
                                       str(local), "--parts", str(args.parts)])
    print("\n完成。冒烟：python eval/run_e4.py --limit-queries 3 --no-reader")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
