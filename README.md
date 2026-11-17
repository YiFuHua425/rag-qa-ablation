# RAG 消融研究 · 专题汇报仓库

> 面向中文问答的检索增强生成：**混合检索与重排策略的消融研究**
> 课程专题汇报（主题编号 **26**）· 全链路**零训练**，只做检索与推理

本仓库是**第 0 棒**交付物：接口冻结 + 空实现骨架。任何人 clone 后都能立刻开工，
不必等别人的模块写完。

---

## 快速开始

```bash
git clone https://github.com/YiFuHua425/rag-qa-ablation.git
cd rag-qa-ablation

# 1) 验证骨架（零依赖，只用标准库，不需要装 torch / faiss）
python eval/run_e1.py --dry-run

# 期望输出：
#   [E1] split=dry  n=30  config=dense_topk5
#   Recall@1 / Recall@5 / MRR / nDCG@5 / F1 / latency
#   → results/raw/E1_dryrun.json
```

看到上面这几行数字，说明**接口是通的**，可以开始接自己的模块了。
`--dry-run` 用的是真实指标代码 + 假检索结果，所以它同时验证了 A 的骨架和 D 的指标。

## 装真实依赖（B / C 需要，A / D / E 不装也能干活）

```bash
python -m venv .venv && .venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

---

## 目录结构

```
.
├── INTERFACE.md          ★ 接口契约，所有人先读这个
├── config.yaml             所有超参，禁止硬编码
├── requirements.txt
├── data/                   语料与问题（jsonl 不进 git，见 .gitignore）
│   ├── corpus.jsonl        {"doc_id","text"}                   ← D 产出
│   └── dev.jsonl           {"qid","question","gold_doc_id",
│                            "answer_text","answer_start"}      ← D 产出
├── retrieval/
│   ├── bm25.py             BM25 打分（B 手写）
│   ├── bge_encoder.py      BGE 编码 + FAISS 索引（B）
│   └── rrf.py              倒数排名融合 RRF（B 手写）
├── rerank/
│   └── cross_encoder.py    交叉编码器重排（C）
├── reader/
│   ├── context_builder.py  上下文拼装与位置控制（C）
│   └── extractive_qa.py    抽取式阅读器（C）
├── eval/
│   ├── metrics.py          Recall / MRR / nDCG / EM / F1 / bootstrap（D 校验扩充）
│   ├── check_data.py       ★ 数据闸门，D 用它做验收
│   ├── run_e1.py           基线入口（含 --dry-run）
│   └── run_exp.py          批量实验入口（E1…E6 的配置矩阵）
├── results/{raw,fig,tab,log}/
├── report/                 报告与图表源
├── ppt/                    演示文稿与讲稿
└── docs/pitfalls.md        ★ 踩坑日志，所有人随时追加
```

---

## 各角色从这里开始

| 角色 | 第一步 | 验收命令 |
|---|---|---|
| **B 检索** | 读 `INTERFACE.md` 的 `BM25` / `BgeEncoder` / `fuse` 签名，先把 `bm25.py` 的 STUB 换成真实现（可用 10 条手写样本自测，不必等数据） | `python eval/run_e1.py --dry-run` 仍全绿 |
| **C 重排/生成** | 换 `rerank/cross_encoder.py`、`reader/*.py` 的 STUB；先让单条 (question, context) → answer 跑通 | `python eval/run_exp.py --exp smoke --dry-run` |
| **D 数据/评测** | 先跑 `python eval/check_data.py`（现在会提示数据缺失）；产出 `corpus.jsonl`/`dev.jsonl` 后必须全绿 | `python eval/check_data.py --strict` |
| **E 写作** | 报告目录写进 `report/`，图表从 `results/fig/` 取；不要自己算指标 | — |

## 分支与合并

- `main` **受保护，不许直推**；每人一条 `feat/<角色>`，例如 `feat/b-retrieval`
- 每晚合并一次；冲突当天解
- 提交信息前缀：`feat:` / `fix:` / `exp:` / `docs:`

## 交接单（每棒结束发群里，四行）

```
【第 X 棒交接】 交棒人：___
交付物：<文件路径 / 命令>
验收方式：<一条能跑的复现命令>
已知限制：<没做完的部分、遗留问题>
```

---

## 实验编号（全组统一口径）

| 编号 | 内容 | 负责人 | 产出 |
|---|---|---|---|
| E1 | 稠密检索基线 | A | `results/raw/E1_*.json` |
| E2 | BM25 / 稠密 / RRF 融合（α 扫描） | B | `fig/E2_alpha_recall.png` |
| E3 | 切块策略 | B | `fig/E3_chunk.png` |
| E4 | 重排与候选深度（含延迟） | C | `fig/E4_pareto.png` |
| E5 | 上下文位置偏差 | C | `fig/E5_position.png` |
| E6 | 查询改写 RM3（加分，可弃） | B | `tab/E6_rm3.csv` |

## 当前进度

- [x] **第 0 棒**：接口冻结 + 空实现 + `--dry-run` 可跑 ← 现在在这里
- [ ] 第 1 棒（D）：数据就绪，`check_data.py --strict` 全绿
- [ ] 第 2 棒（A+B+C）：打通一条直线，出第一个 `Recall@5`
- [ ] 第 3 棒（B/C）：三件套可开关（6 种配置）
- [ ] 第 4 棒（全员）：E1–E4 矩阵 + 主结果总表
- [ ] 第 5 棒（C）：E5 位置偏差
- [ ] 第 6 棒（B，可弃）：E6 查询改写
- [ ] 第 7 棒（E）：报告 / PPT / 录屏 / 提交