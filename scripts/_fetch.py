"""单流下载小工具（modelscope 不支持 HEAD，分段逻辑用不了时用它）。

    python scripts/_fetch.py <url> <out_path>

只做一件事：GET → 写文件，每 1 MB 打一次进度。断点续传不支持（换 URL 就重新下）。
"""
import sys
import time
import urllib.request
from pathlib import Path

url, out = sys.argv[1], Path(sys.argv[2])
out.parent.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0"}
t0 = time.time()
done = 0
with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=900) as r, \
        open(out, "wb") as fh:
    total = int(r.headers.get("Content-Length") or 0)
    while True:
        buf = r.read(1 << 20)
        if not buf:
            break
        fh.write(buf)
        done += len(buf)
        print(f"\r{done / 1e6:7.1f}/{total / 1e6:.1f} MB "
              f"{done / 1e6 / max(1e-9, time.time() - t0):5.2f} MB/s", end="", flush=True)
print(f"\n[ok] {out} {out.stat().st_size / 1e6:.1f} MB")
