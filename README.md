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
│   ├── build_data.py       CMRC2018 → corpus/dev（D，可复现）
│   ├── corpus.jsonl        {"doc_id","text"}                   ← D 产出
│   ├── dev.jsonl           {"qid","question","gold_doc_id",
│   │                        "answer_text","answer_start"}      ← D 产出
│   └── dev_smoke.jsonl     500 题子集（seed=42）                 ← D 产出
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

- [x] **第 0 棒**：接口冻结 + 空实现 + `--dry-run` 可跑
- [x] **第 1 棒（D）**：数据就绪，`check_data.py --strict` 全绿
  （3507 段 / 3219 题 / smoke 500 题 / 答案越界 0；见 `data/README.md`）
- [x] **第 2 棒（A+B+C）**：直线打通 —— B 的检索 + C 的阅读器，`python eval/run_e1.py --split smoke`
  出 Recall@1/5、MRR、F1（阅读器不再是 STUB，F1/EM 是真数字）
- [x] **第 3 棒（B/C）**：三件套可开关 —— `python eval/run_exp.py --exp smoke` 6/6 全绿
  （重排已接进同一入口，见 `eval/run_exp.py` 的 rerank 分支）
- [x] **第 4 棒（全员）**：E1–E4 矩阵 —— C 的 E4 已出全指标 + 延迟 + `fig/E4_pareto.png`
  （R@1 0.880→0.970、MRR 0.922→0.978、F1 0.814→0.834；depth 20 饱和）
- [x] **第 5 棒（C）**：E5 位置偏差 —— `fig/E5_position.png` + 结论
  （4000 字档：首 0.840 / 自然 0.834 / 尾 0.731 / 中 0.718；2000 字档尾部保留率仅 0.10、F1 0.212）
- [x] **第 6 棒（B，加分）**：E6 查询改写 —— 负结果如实报告
  （RM3+RRF −0.06 点 / RM3+BM25 −1.74 点，见 `tab/E6_rm3.csv`）
- [~] **第 7 棒（E）**：报告 / PPT / 录屏 / 提交 —— A、D 负责的章节已完成，见下

## 报告交付物导航

| 文件 | 内容 | 负责人 | 状态 |
|---|---|---|---|
| `report/00_目录与章节分工.md` | 20 页预算 + 每章归属 + 图表编号 | A | ✅ |
| `report/分工表.md` | 五人角色、棒次、逐人交付（可直接进报告与 PPT） | A | ✅ |
| `report/主结果总表.md` | 口径说明 + 10 条可直接引用的结论 + 三张对比表 | A | ✅ |
| `results/tab/主结果总表.csv` | E1–E6 共 29 行，含 split/n/CI/延迟/来源 run_id | A | ✅ |
| `report/第2章_相关工作.md` | BM25 → DPR → ColBERT → RRF → RAG | B | ✅ |
| `report/第3章_算法原理.md` | 形式化 + 五个组件的公式与取舍 | **A** | ✅ |
| `report/第4章_实现过程.md` | 环境 / 数据 / 超参 / 6 个实现细节 / 踩坑 / 复现 | **D** | ✅ |
| `report/附录B_检索实验记录.md` | E2 / E3 / E6 明细 | B | ✅ |
| `report/附录C_重排与位置实验记录.md` | E4 / E5 明细 | C | ✅ |
| `results/fig/架构图.png` | 图 3-1 系统架构与数据流 | A | ✅ |
| 第 1、5、6 章 + 结论 + 参考文献 + 统稿 + PPT + 录屏 | — | **E** | ⏳ |

> 给 E 的入口：先读 `report/00_目录与章节分工.md`（页数预算与压缩顺序），
> 第 5 章的数字**只从** `report/主结果总表.md` 取，不要自己算。

## 单题链路演示（`demo.py`）

一条命令把「检索 → 融合 → 重排 → 拼装 → 抽取」全链路摊开，**用于录屏与 PPT 截图**。
自动挑一题「一路没召回 / 融合有提升」的样本，让每一步的差异看得见。

```powershell
python demo.py                      # 自动选题（默认 smoke，约 5 秒，缓存已暖时）
python demo.py --alpha 0.2          # 用 E2 扫出的最优融合权重
python demo.py --index 3            # 指定某题
python demo.py --question "苹果公司的创始人是谁"   # 自定义问题（无金标）
python demo.py --list               # 列出可演示的题目
```

输出分 9 段：装配 → 分词 → 稀疏路 → 稠密路 → RRF → （可选）重排 → 上下文拼装 → 阅读器 → 小结。
小结那张表就是全篇的论证结构：同一个金标在 BM25 / 稠密 / 融合 / 重排 下各排第几。

依赖分层：**检索侧只需 numpy + onnxruntime + jieba**；阅读器与重排需要 torch + 权重，
缺失时不报错、只打印「跳过 + 怎么装」。

## 成员与 GitHub 账号对照

| 成员 | 角色 | GitHub 账号 | 提交数 |
|---|---|---|---|
| 颜福洪 | A 系统集成与协调 | `YiFuHua425` | 13 |
| 龙新月 | B 检索与融合 | `FUYU0v0` | 2 |
| 王文超 | C 重排与阅读器 | `yytwwg` | 5 |
| 孔得地 | D 数据与评测 | `xiao-ning77` | 1 |
| 袁毅 | E 报告与呈现 | `122322-xx` | 0（负责文档/PPT，不产生代码提交） |

> 提交数为 2026-11-23 快照。C 的 5 个提交早期署名为占位值 `C <c@example.com>`
> （当时他没有写权限，是以 bundle/patch 离线交付的），现由仓库根的 `.mailmap`
> 归因到真实账号。本地验证：`git log --use-mailmap --format='%aN' HEAD`。