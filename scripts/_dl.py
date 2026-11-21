"""多连接分段下载（支持断点续传），本机镜像站单连接只有几十 KB/s 时用这个。

用法：
    python scripts/_dl.py <url> <out_path> [--parts 4] [--insecure]

行为：
- 先 HEAD 拿 Content-Length，按 --parts 切成若干段并发下载到 <out>.partN
- 已存在的段会跳过（续传）；全部完成后拼成 <out> 并删除分段
- 只对支持 Range 的服务器生效；不支持则退化为单连接
"""
from __future__ import annotations

import os
import ssl
import sys
import threading
import time
import urllib.request
from pathlib import Path

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

UA = {"User-Agent": "Mozilla/5.0"}


def head(url: str) -> tuple[int, bool]:
    req = urllib.request.Request(url, method="HEAD", headers=UA)
    with urllib.request.urlopen(req, timeout=30, context=CTX) as resp:
        size = int(resp.headers.get("Content-Length") or 0)
        ranges = resp.headers.get("Accept-Ranges", "")
    return size, "bytes" in ranges.lower()


def fetch_range(url: str, start: int, end: int, out: Path, stop: threading.Event,
                progress: dict, key: int):
    existing = out.stat().st_size if out.exists() else 0
    total = end - start + 1
    if existing >= total:
        progress[key] = total
        return
    headers = dict(UA)
    headers["Range"] = f"bytes={start + existing}-{end}"
    for attempt in range(6):
        if stop.is_set():
            return
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60, context=CTX) as resp, open(out, "ab") as fh:
                while True:
                    if stop.is_set():
                        return
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    fh.write(chunk)
                    progress[key] = (out.stat().st_size)
            if out.stat().st_size >= total:
                return
        except Exception as exc:  # noqa: BLE001
            print(f"  [part{key}] 重试 {attempt + 1}: {type(exc).__name__}", flush=True)
            time.sleep(2 + attempt * 3)
    print(f"  [part{key}] 放弃", flush=True)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    url, out_path = sys.argv[1], Path(sys.argv[2])
    parts = 4
    if "--parts" in sys.argv:
        parts = int(sys.argv[sys.argv.index("--parts") + 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    size, supports = head(url)
    print(f"[dl] {url}\n[dl] size={size / 1e6:.1f} MB ranges={supports} parts={parts}", flush=True)
    if not size:
        return 1
    if not supports:
        parts = 1

    step = size // parts
    bounds = []
    for i in range(parts):
        start = i * step
        end = size - 1 if i == parts - 1 else (i + 1) * step - 1
        bounds.append((start, end))

    part_files = [out_path.with_suffix(out_path.suffix + f".part{i}") for i in range(parts)]
    progress = {i: (pf.stat().st_size if pf.exists() else 0) for i, pf in enumerate(part_files)}
    stop = threading.Event()
    threads = []
    for i, (s, e) in enumerate(bounds):
        t = threading.Thread(target=fetch_range, args=(url, s, e, part_files[i], stop, progress, i),
                             daemon=True)
        t.start()
        threads.append((t, e - s + 1))

    t0 = time.time()
    while any(t.is_alive() for t, _ in threads):
        done = sum(min(progress[i], tot) for i, (_, tot) in enumerate(threads))
        rate = done / max(1e-9, time.time() - t0) / 1024
        print(f"\r[dl] {done / 1e6:6.1f}/{size / 1e6:.1f} MB  {rate:7.1f} KB/s", end="", flush=True)
        time.sleep(1)
    stop.set()
    print()

    done = sum(min(progress[i], tot) for i, (_, tot) in enumerate(threads))
    if done < size:
        print(f"[dl] 未完成 {done}/{size}，分段文件保留可续传")
        return 2

    with open(out_path, "wb") as dst:
        for pf in part_files:
            with open(pf, "rb") as src:
                while True:
                    buf = src.read(1 << 20)
                    if not buf:
                        break
                    dst.write(buf)
            pf.unlink()
    print(f"[dl] 完成 → {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
