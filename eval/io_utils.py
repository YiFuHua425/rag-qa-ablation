"""实验脚本的公共设施：读配置、读数据、写结果、记日志 —— 负责人：B

为什么单独一个文件：``eval/run_e2.py`` / ``run_e3.py`` / ``run_e6_rm3.py``
要写同一套 ``results/raw/<run_id>.json``（格式见 INTERFACE.md 第 3 节），
格式一处定义、三处复用，避免"三个脚本三种写法"。

路径与字段全部来自 ``config.yaml``，脚本里不出现魔法数字。
"""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"


# ---------------------------------------------------------------- 配置
def load_config(path: str | Path | None = None) -> dict:
    """读 ``config.yaml``。装不上 pyyaml 时给出明确提示，而不是静默用默认值。"""
    path = Path(path) if path else CONFIG_PATH
    try:
        import yaml
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"读 {path.name} 需要 pyyaml（requirements.txt 已登记）：pip install pyyaml"
        ) from exc
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    return cfg


def _p(cfg: dict, dotted: str, default=None):
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def path_of(cfg: dict, key: str) -> Path:
    """``cfg["paths"][key]`` → 绝对路径。"""
    rel = _p(cfg, f"paths.{key}", key)
    return (ROOT / rel).resolve()


# ---------------------------------------------------------------- 环境信息
def git_commit() -> str:
    """拿当前 commit；克隆下来的仓库里没有 .git 时返回 ``"unknown"``。"""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=str(ROOT), capture_output=True, text=True, timeout=3)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return os.environ.get("GIT_COMMIT", "unknown")


def runtime_info() -> dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for mod, key in (("numpy", "numpy"), ("onnxruntime", "onnxruntime"),
                     ("faiss", "faiss"), ("torch", "torch"), ("jieba", "jieba")):
        try:
            m = __import__(mod)
            info[key] = getattr(m, "__version__", "ok")
        except Exception:
            info[key] = None
    return info


# ---------------------------------------------------------------- 数据
def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_split(split: str = "full", cfg: Optional[dict] = None) -> Tuple[List[dict], List[dict]]:
    """读 ``data/corpus.jsonl`` 与 ``data/dev.jsonl``，返回 ``(docs, queries)``。

    ``split="smoke"`` 时优先用 ``data/dev_smoke.jsonl``（check_data.py 生成，
    固定 seed），没有就按 ``project.seed`` 现场抽 ``data.smoke_size`` 题。
    """
    cfg = cfg or load_config()
    corpus_path = ROOT / _p(cfg, "data.corpus", "data/corpus.jsonl")
    dev_path = ROOT / _p(cfg, "data.dev", "data/dev.jsonl")
    if not corpus_path.exists() or not dev_path.exists():
        raise SystemExit(
            f"缺少数据文件：\n  {corpus_path}\n  {dev_path}\n"
            "先跑 `python scripts/build_data_cmrc.py`（或由 D 同学提供 corpus.jsonl / dev.jsonl）"
        )
    docs = read_jsonl(corpus_path)
    queries = read_jsonl(dev_path)

    if split == "smoke":
        smoke_path = ROOT / "data" / "dev_smoke.jsonl"
        if smoke_path.exists():
            queries = read_jsonl(smoke_path)
        else:
            n = int(_p(cfg, "data.smoke_size", 500))
            seed = int(_p(cfg, "project.seed", 42))
            queries = random.Random(seed).sample(queries, min(n, len(queries)))
    elif split != "full":
        raise SystemExit(f"未知 split {split!r}（只支持 smoke / full）")
    return docs, queries


# ---------------------------------------------------------------- 结果落盘
def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Logger:
    """同时写终端与 ``results/log/<run_id>.log``。"""

    def __init__(self, run_id: str, cfg: Optional[dict] = None) -> None:
        cfg = cfg or {}
        log_dir = path_of(cfg, "results_log") if cfg else ROOT / "results" / "log"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / f"{run_id}.log"
        self._fh = self.path.open("w", encoding="utf-8")

    def __call__(self, msg: str = "") -> None:
        print(msg, flush=True)
        self._fh.write(msg + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def write_run(run_id: str, *, config: dict, metrics: dict, per_question: Sequence[dict],
              meta: Optional[dict] = None, cfg: Optional[dict] = None) -> Path:
    """按 INTERFACE.md 第 3 节写 ``results/raw/<run_id>.json``。"""
    cfg = cfg or {}
    raw_dir = path_of(cfg, "results_raw") if cfg else ROOT / "results" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    base_meta = {
        "created_at": _now(),
        "seed": int(_p(cfg, "project.seed", 42)),
        "git_commit": git_commit(),
        "python": platform.python_version(),
    }
    base_meta.update(meta or {})
    payload = {
        "run_id": run_id,
        "config": config,
        "meta": base_meta,
        "metrics": metrics,
        "per_question": list(per_question),
    }
    path = raw_dir / f"{run_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def write_table(name: str, rows: Sequence[dict], cfg: Optional[dict] = None) -> Path:
    """写 ``results/tab/<name>.csv``（标准库 csv，不引入 pandas）。"""
    import csv

    cfg = cfg or {}
    tab_dir = path_of(cfg, "results_tab") if cfg else ROOT / "results" / "tab"
    tab_dir.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        raise SystemExit("write_table 收到空表")
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    path = tab_dir / f"{name}.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: _round(r.get(k)) for k in fields})
    return path


def _round(v):
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(str(_round(x)) for x in v) + "]"
    return v


def dump_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def add_repo_to_path() -> None:
    """允许 ``python eval/run_e2.py`` 与 ``python -m eval.run_e2`` 两种跑法。"""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
