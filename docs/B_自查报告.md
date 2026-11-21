# 分工 B 自查报告（按标准逐条核对 + 数据复核）

> 自查时间：2026-11-21
> 自查人：B（检索）
> 自查依据（仓库里 5 份文档，全部通读）：`INTERFACE.md`、`README.md`、`data/README.md`、
> `执行顺序-接力棒与交接标准.md`、`执行计划-4天时间线与任务顺序.md`、`专题汇报方案-检索增强生成.md`
> 以及课程要求（`专题汇报要求.pptx`）。

---

## 0. 三句话结论

1. **标准里点名 B 的每一条都有落点、都能跑、都有证据**（见第 1 节的核对表），
   `INTERFACE.md` 的 3 个冻结签名一字未改（只加了**带默认值的可选参数**，位置参数顺序不变）。
2. **有 3 处碰到了别人负责的文件**：`eval/run_e1.py`（A）、`eval/run_exp.py`（A）、
   `eval/plots.py` + `results/fig/*.png`（第 4 棒写的是 D「出版式统一的所有图」）。
   这 3 处都是「不写就交不出判据」的地方，属于**必须由 A / D review 后合入**的灰色地带，
   详见第 2 节——我没有悄悄改，全部在补丁包里单列并写明理由。
3. **数据自查通过**：字段、唯一性、金标存在性、答案子串、`answer_start` 精确性全部 0 错；
   并且我从**原始 squad-style json 独立重算**了一遍，3507 段 / 3219 题 / 3 题位置校正与 D 的口径完全一致（见第 3 节）。
4. **自查过程中又抓出 8 个问题并全部修掉**（第 4 节）：仓库里混进的垃圾文件、无用的 `pyarrow` 依赖、
   3 处硬编码 `k=60`、结果 JSON 不记用的是 int8 还是 fp32、`per_question` 缺 `pred`/`f1`、
   量化脚本的编码耗时被缓存吞掉、后端对照默认没钉权重、`run_e1` 写死 numpy 后端。
   修完全量重跑，**主结论数字一个都没变**。

---

## 1. 逐条核对（标准原文 → 我做了什么 → 证据）

### 1.1 `INTERFACE.md`

| 条款 | 要求 | 落实 | 证据 |
|---|---|---|---|
| §1 数据格式 | `corpus.jsonl` 只有 `doc_id/text`；`dev.jsonl` 5 字段；`answer_text` 必须是金标段子串、越界 0 | 只读不造；D 的产物额外带 `source_id`（INTERFACE 11-21 变更记录已授权） | `eval/check_data.py --strict` 四项断言全过 |
| §1 评测两档 | `smoke` = 500 题 seed=42；`full` = 3219 题 | smoke 用 D 的 `data/dev_smoke.jsonl`（重生成字节一致），full 用 `dev.jsonl` 全集 | `E2_*.json` meta `n_questions=3219`；`E1_dense_top5.json` `n_questions=500` |
| §2 `BM25` | `__init__(k1=1.5, b=0.75)`、`fit(docs)`、`search(query, topk) -> [(doc_id, score)]`；公式与 IDF 按原文；`jieba.lcut`；查询词去重；按分降序截断 | 签名完全一致（额外可选 `tokenizer=`、`dedup_query=True`，不破坏位置调用）；`retrieval/tokenization.py` 用 `jieba.lcut`；`dedup_query=True` | `retrieval/bm25.py`；`tests/test_retrieval.py` 用单文档手算对齐公式；`pytest` 13 passed |
| §2 `BgeEncoder` / `FaissIndex` | `encode_docs(docs)->(N,d)`、`encode_query(q)->(d,)`、`build(doc_ids, embs)`、`search(q_emb, topk)->[(doc_id,score)]`；L2 归一化 + `IndexFlatIP`；doc 侧**必须落盘缓存**；模型名走 config；`_model` 懒加载 | 签名一致；编码后 assert 范数≈1；`FaissIndex(index_type="IndexFlatIP")`，faiss 可用时真建 `IndexFlatIP`，不可用时 numpy 内积（同一数学）；缓存 `.cache/emb/emb_<后端>-<权重文件>-<max_len>-<norm/raw>-<语料sha1>_n<N>.npz`；模型名取 `config.yaml`；`_ensure_model()` 懒加载、`--dry-run` 不触发 | `retrieval/bge_encoder.py`；`E2_*.json` config 里 `model` / `model_file` / `index_backend` 三个字段可追溯 |
| §2 `fuse` | `fuse(runs, weights=None, k=60)`；`score=Σ w_i/(k+rank_i)`，`rank` 从 1 起；缺失路贡献 0；weights 缺省等权；α=稠密权重、1−α=BM25 权重 | 完全按此实现；同路重复 doc_id 只算首现；`k<=0` 抛 `ValueError`；weights 长度不符抛 `ValueError` | `retrieval/rrf.py`；`tests` 里 5 条断言覆盖 |
| §3 结果格式 | `results/raw/<run_id>.json` 五段结构；`run_id` 形如 `E2_rrf_alpha0.5`；图 `fig/E2_alpha_recall.png`；表 `tab/E1_baseline.csv` | 五段齐全；run_id 命名与 `run_exp.MATRIX` 对齐（`E2_bm25`/`E2_rrf_alpha{0.0…1.0}`/`E3_chunk_para|256|512`）；图名与 README 表格逐字一致 | 26 个 `results/raw/*.json`、6 张 `results/tab/*.csv`、4 张 `results/fig/*.png` |
| §4 配置 | 所有超参走 `config.yaml`，**禁止硬编码** | k1/b/k/depth/topk/bootstrap_n/seed/max_chars/encode_batch 全部读 `config.yaml`；**自查发现 3 处硬编码 `k=60`（`run_extras_b` / `run_quant_check` / `run_backend_check`），已改成读配置** | 见第 4 节第 3 条 |
| §4 种子 | `seed=42`，bootstrap 固定 | 全部 `seed=cfg.project.seed`（42）；`bootstrap_ci(..., seed=42)` | 各 run json `meta.seed=42` |
| §4 依赖 | 新增依赖必须进 `requirements.txt` | 追加 `onnxruntime>=1.16`、`pytest>=7.4`（**自查发现 `pyarrow` 已无用，已删**，因为数据改由 D 的纯标准库脚本产出） | `requirements.txt` 只增不删原有行 |
| §4 日志 | 真实跑实验写 `results/log/<run_id>.log` | 写 `E2_run.log`/`E3_run.log`/`E6_run.log`（**按脚本名而非 run_id**，属于轻偏差，已记入第 5 节） | `results/log/` |
| §4 指标 | 只调 `eval/metrics.py` | `eval/retrieval_eval.py` 是唯一聚合层，内部只调 `M.recall_at_k/mrr/ndcg_at_k/bootstrap_ci`，无第二套实现 | `eval/retrieval_eval.py`；`tests` 里有「与 metrics 数值一致」断言 |
| §4 大文件 | 数据与 `results/raw` 不进 git | `data/*.jsonl`、`data/raw/`、`results/raw/*.json` 均被原 `.gitignore` 挡住；**我另加 `models/.gitignore` 挡掉 `*.onnx`**（22.8MB + 94.9MB，GitHub 单文件上限 100MB） | `.gitignore` + `models/.gitignore` |

### 1.2 `README.md`

| 要求 | 落实 |
|---|---|
| `main` 受保护、每人 `feat/<角色>` 分支、提交前缀 `feat:/fix:/exp:/docs:` | 一条都没推；推送说明写死 `feat/b-retrieval`（小写，照 README 的例子）+ PR |
| 交接单四行（交付物 / 验收方式 / 已知限制 / 下一棒） | `docs/B_交接单.md` 第 1–4 节，每棒一张四行单 |
| 实验编号归属：E2/E3/E6 归 B，图名固定 | 三个实验都有 json + csv + 图，文件名逐字对齐 |
| B 的第一步验收 `python eval/run_e1.py --dry-run` 仍全绿 | 改动后仍全绿（`--dry-run` 分支一行未动） |

### 1.3 `execution 接力棒`（B 横跨四棒）

| 棒 | 原文要求 | 落实 | 判据实测 |
|---|---|---|---|
| 第 2 棒 | BGE 编码全库 + FAISS 建索引 + topk，单查询 < 1 秒 | 3507×512 编码 53.8 s（int8）；索引 `IndexFlatIP`；端到端 1.4–3.5 ms | `run_e1.py --split smoke` → R@1 0.884 / **R@5 0.934** / MRR 0.907 / 1.42 ms ✓（自检线 0.85–0.96） |
| 第 3 棒 | 手写 BM25、手写 RRF、开关 `--retriever dense\|bm25\|hybrid` + `--alpha` | `bm25.py`/`rrf.py` 均为手写（无 pip 实现）；`run_exp.py` 有 `cmd_real()`；`run_e3.py --retriever` 接受 `dense/bm25/hybrid/rrf` | `run_exp.py --exp smoke` → `smoke_dense/bm25/rrf_a0.5/rrf_a1.0` 四条落盘，2 条 rerank skip（C 的 STUB） |
| 第 4 棒 | E2 七个 α 点、E3 三档切块（+我另加四档） | 7 个 `E2_rrf_alpha*` + `E2_bm25` + `E2_dense`；`E3_chunk_{para,256,512}` + 四档扩展 | α=0.2 最优 R@5 **0.985399** [0.981361,0.989394]；重跑两遍逐位一致（远小于 0.1 点） |
| 第 6 棒 | RM3 与最优配置对比，正负都写 | `run_e6_rm3.py` 四条 run | BM25 路 −1.74pt、融合路 −0.06pt（负结果，已写入附录 B.5） |
| 第 7 棒 | A/B/C/D 各供自己那段分析 + PPT 自己的页 | `report/第2章_相关工作.md`、`report/附录B_检索实验记录.md`、5 页 PPT + 讲稿 | docx 252 段 / 13 表；PPT 几何自检无越界 |

### 1.4 课程要求（`专题汇报要求.pptx`）

| 要求 | 落实 |
|---|---|
| 报告正文宋体小四、正文 ≤ 20 页 | `report/分工B_报告章节.docx`：宋体 12pt（小四）、1.5 行距、首行缩进 2 字符；第 2 章 5332 字 ≈ 5.3 页 + 附录 7190 字 ≈ 7.2 页（合计≈12.5 页，**并组后需 E 统稿压到 20 页内**） |
| 报告须含「调研/原理/实现/实验/思考」 | 第 2 章=调研；模块说明=原理；附录 B.6=实现问题；B.2–B.5=实验；B.7=思考与改进 |
| PPT 与报告均注明小组分工 | 每页 PPT 有「分工 B ｜ …」角标；两份 md 首行标注「分工 B」 |

---

## 2. 越界 / 灰色地带（**需要 A、D 确认**，我没有偷偷改）

| # | 文件 | 谁的 | 我做了什么 | 为什么不得不动 | 建议 |
|---|---|---|---|---|---|
| 1 | `eval/run_e1.py` | **A**（第 2 棒原文：「A：把 B 和 C 串起来…写 `run_e1.py`」） | 只填 `run_real()` + 4 个新 CLI 参数；`--dry-run` 与骨架逻辑一行未动（diff 可查） | 第 2 棒判据 `python eval/run_e1.py --split smoke` 必须能跑；A 侧没接线上，B 的检索就没有对外入口 | **保留但请 A 过一遍**；若 A 自己写，直接 `git checkout -- eval/run_e1.py` 回退，我的检索仍可被 `eval/run_e2.py` 独立调用 |
| 2 | `eval/run_exp.py` | **A**（骨架） | 只补 `cmd_real()`（真跑分支）+ 3 个参数；`--list`/`--dry-run`/`MATRIX` 未动 | 第 3 棒原文点名 B：「接进主流程，加开关 `--retriever`/`--alpha`」，判据就是 `run_exp.py --exp smoke` | 同上，属分工原文授权的改动 |
| 3 | `eval/plots.py` + `results/fig/*.png` | **D**（第 4 棒原文：「D：所有 run 统一算指标 + bootstrap + 出版式统一的所有图」） | 写了 4 张图（E2 α 曲线、E2 延迟、E3 切块、E6 RM3） | README 的实验编号表把 `fig/E2_alpha_recall.png`、`fig/E3_chunk.png` 列在 **B 的产出**下，但第 4 棒把「出版式统一的图」给了 D | **保留为由 D 覆盖的草稿**：D 若要统一配色/字号/字体，直接改 `eval/plots.py` 重出即可，B 不占坑 |
| 4 | `docs/pitfalls.md` | 全员共享 | 填了「（B 补）」的 4 条 + 追加若干条 | 文件本来就写给 B 补 | 已把 D 的那条 matplotlib 待记项改成「原 D 待记项的实测答案」，**不占 D 的账** |
| 5 | `requirements.txt` | 全员共享 | 末尾追加 onnxruntime / pytest | INTERFACE 要求新依赖必须登记 | 只增不改；已删掉自己加的 `pyarrow`（现在用不到） |

### 2.1 标准里没要求、我自己加的（可整段丢弃，不影响验收）

| 加了什么 | 位置 | 性质 | 处理建议 |
|---|---|---|---|
| 13 项单元测试 | `tests/test_retrieval.py` | 自检，不影响任何人 | 保留（CI 友好）；不想留就删掉整个文件 |
| 超参/量化/后端三个补充实验 | `eval/run_extras_b.py`、`run_quant_check.py`、`run_backend_check.py` | 对应方案「有改进实验可加分」+ pitfalls 里 B 的待记项 | 保留；报告里作为「改进实验」，与主表分开 |
| md → docx 排版脚本 | `scripts/md_to_docx_b.py` | 只是把我的 md 变成能直接交的 docx | 保留（E 统稿时可能用不上） |
| PPT 生成 + 几何自检 | `scripts/make_ppt_b.py`、`inspect_ppt_b.py`、`check_text_fit.py` | 5 页 B 片段 | 保留 |
| 第 2 章相关工作代写 | `report/第2章_相关工作.md` | 方案把「第 2 章 相关工作」明确分给 B | 保留 |
| 打包脚本 | `scripts/make_push_bundle.py` | 交付辅助 | 保留 |

---

## 3. 数据自查（不看 D 的结论，自己独立算一遍）

### 3.1 结构自查（直接读 jsonl）

```
corpus.jsonl  3507 行 | doc_id 唯一 ✔ | doc_id 连续 p_00001…p_03507 ✔ | text 无重复、无空 ✔
dev.jsonl     3219 行 | qid 唯一 ✔ | qid 连续 dev_0001…dev_3219 ✔
                        gold_doc_id 全部存在于 corpus ✔（0 条缺失）
                        answer_text 全部是金标段落子串 ✔（0 条越界）
                        answer_start 指向位置与 answer_text 完全一致 ✔（0 条不符）
                        空答案 0 / 空问题 0 ✔ | source_id 全有且唯一 ✔
dev_smoke.jsonl 500 行 | 是 dev 的子集 ✔ | 用 --make-smoke 500 重生成，**字节完全一致**（seed=42 可复现）✔
```

### 3.2 独立重算（从原始 squad-style json 重算，不读 D 的脚本输出）

```
原始段落去重后           3507   = train 2403 + dev 848 + trial 256 ✔（与 D 的 data/README.md 一致）
原始 dev 题数            3219   ✔（全部保留，0 丢弃）
answer_start 需校正题数     3   ✔（与 D 的 build_stats 一致；全部用 context.find 重算）
段落长度                 均值 509.1 字 / 最长 980 字 ✔
答案长度                 均值 10.7 字 ✔
```

### 3.3 唯一发现的瑕疵（保留，但要在报告里说明）

- **1 组重复问题文本**：`dev_1731`（`source_id=DEV_519_QUERY_0`）与 `dev_1748`（`DEV_525_QUERY_0`）
  都是「雅芳河的发源地是哪里？」，但**金标段落不同**（`p_02887` vs `p_02891`）。
  这是 CMRC2018 原始数据里就有的两条不同条目，不是造数错误。
  **处理：两条都保留**（丢掉任何一条都等于篡改 D 的数据口径），在报告的「数据说明」里注明一句即可。

### 3.4 编号与缓存的一致性（上一轮踩过坑的地方）

D 的编号从 `p_00001` 起、语料顺序 train→dev→trial，与我早期自造数据（`p_00000` 起）完全不同。
向量缓存若只按「块数」做键，会**静默命中旧向量**。已修：缓存键含语料 sha1 摘要，npz 内也存摘要、命中时再核一次。
重跑验证：换数据后 E1/E2/E3/E6/超参/量化/后端所有指标**逐位不变**。

---

## 4. 自查中发现并已修掉的问题（本轮）

1. **仓库里混进垃圾**：`.pytest_cache/`（4 个文件）、已过时的 `scripts/{_fetch,_probe_wheels,download_cmrc}.py`、
   `data/raw/*.parquet`（4.9MB，是早期 parquet 路线残留，D 的脚本用 json）、`results/raw/E1_dryrun.json` → **已全部删除**。
2. **`requirements.txt` 里 `pyarrow` 已无用途**（数据改为 D 的纯标准库脚本产出）→ 已删，并写明原因。
3. **三处硬编码 `k=60`**（`run_extras_b.py` / `run_quant_check.py` / `run_backend_check.py`）→ 改为读
   `config.yaml → retrieval.fusion.k`（违反 INTERFACE §4「禁止硬编码」）。
4. **结果 JSON 缺可追溯字段**（最要紧的一条）：原来只记 `model: BAAI/bge-small-zh-v1.5`，
   但目录里同时有 `model.onnx`（fp32）与 `model_int8.onnx`（int8），**两者单路稠密差 0.44 个点**——
   等于事后无法从结果文件判断「这份数字是哪个权重跑出来的」。已给 E1/E2/E3/E6/补充实验全部补上
   `model_file`（`model_int8.onnx`）与 `index_backend`（`faiss`/`numpy`）。
5. **结果 JSON 字段与 INTERFACE §3 不一致**：`per_question` 缺 `pred`/`f1`、`metrics` 缺 `em`/`f1`
   （检索类 run 不跑阅读器）→ 已显式补 `pred=""`、`f1=null`、`em=null`，不再「缺键」。
6. **`docs/pitfalls.md` 我占了 D 的待记项** → 已把该行标注为「原 D 待记项的实测答案」。
7. **量化脚本的「编码耗时」被缓存吞掉了**（重跑时才发现）：`build_from_encoder` 命中磁盘缓存时只花 0.1–0.3 s，
   于是表里出现「fp32 编码 0.3 s / 32.5 docs·s⁻¹」这种自相矛盾的数。**已改成直接全库真编码再手工建索引**，
   现在的数字是整库实打实的：fp32 **100.0 s**、int8 **53.8 s**（约 1.9×），与附录 B.4.6 一致。
8. **后端对照默认没钉权重**：`run_backend_check.py` 的 `--model-path` 默认 `None`，
   而目录里 fp32 与 int8 并存时后端探测**优先 fp32**，于是表里稠密 R@5 变成 0.9531（fp32 的数），
   与正文「正式口径 int8」对不上。**已改成默认钉 `model_int8.onnx`**（不存在时才回退 auto），
   并把 `eval/run_e1.py` 的索引后端从写死 numpy 改成 `use_faiss=True`（第 2 棒原文要求「FAISS 建索引」）。
   现在两表一致：faiss **0.201 ms** / numpy **0.268 ms**，两后端稠密 R@5 都是 0.9487、RRF 都是 0.9854。

> 第 4/7/8 条改动后重跑了全部实验：**主结论数字一个都没变**（见 `results/tab/*.csv`），
> 变的只有 JSON 里的可追溯字段与两处测量口径（编码耗时、后端延迟）。

---

## 5. 仍然没闭环的地方（诚实清单）

| # | 事项 | 原因 | 谁能补 |
|---|---|---|---|
| 1 | 第 2 棒判据里的 **F1** 是 `null` | `reader/extractive_qa.py` 仍是 C 的 `[STUB]`；B 不碰 C 的模块 | C（改完后 `run_e1.py` 自动填上，无需改代码） |
| 2 | 第 3 棒 **6 条 smoke 配置只绿了 4 条** | 2 条带 rerank，`rerank/cross_encoder.py` 仍是 C 的 `[STUB]` | C |
| 3 | **E3 七档只在 int8 下跑过** | fp32 只做了 E2 口径对照（结论一致），切块网格重跑≈7 分钟，排期不够 | 有空时 `--model-path model.onnx` 重跑一次即可 |
| 4 | 绝对指标偏高（纯 BM25 R@5 就有 0.976） | CMRC2018 的段落与问题同源、段落短（均值 509 字），难度天然偏低 | 报告里说明「看相对差值与 CI」，另外 OOD 数据（SQuAD v2）留给改进 |
| 5 | 融合只做了 RRF，没和 CombSUM / 学习式融合比 | 方案把融合定在 RRF，其他是对照 | 列为「改进」，不影响验收 |
| 6 | PPT/docx 没做视觉渲染 | 本机无 LibreOffice / PowerPoint | 用户用 Office 打开看一眼（尤其 PPT 第 4、5 页表格区） |
| 7 | `results/log/` 按脚本名而不是 run_id 命名 | 一个脚本产出多个 run，共用一份日志更可读 | 若 D 汇总时要求 `<run_id>.log`，我可以逐 run 拆 |

---

## 6. 复现我自己这一套（一条条都能重来）

```powershell
$ws  = "D:\dsh\写报告的地方"
$py  = "D:\dsh\.dsh\dsh-runtimes\dsh-next-primary-runtime\dependencies\python\python.exe"
$env:PYTHONPATH   = "$ws\.pylibs;$ws\rag-qa-ablation"
$env:MPLCONFIGDIR = "$ws\.mplcache"
Set-Location "$ws\rag-qa-ablation"

# 0) 数据（第 1 棒 D 的脚本；原始 json 放 data/raw/）
python data/build_data.py --sample 20
python eval/check_data.py --strict
python eval/check_data.py --make-smoke 500

# 1) 自检
python -m pytest tests -q                    # 13 passed

# 2) 第 2 棒判据（E1 稠密基线）
python eval/run_e1.py --split smoke --model-path models/bge-small-zh-v1.5-onnx/model_int8.onnx

# 3) 第 3 棒判据（配置矩阵）
python eval/run_exp.py --exp smoke --split smoke --model-path models/bge-small-zh-v1.5-onnx/model_int8.onnx

# 4) E2 / E3 / E6
python eval/run_e2.py --split full --model-path models/bge-small-zh-v1.5-onnx/model_int8.onnx
python eval/run_e3.py --split full --grid extended --model-path models/bge-small-zh-v1.5-onnx/model_int8.onnx
python eval/run_e6_rm3.py --split full --model-path models/bge-small-zh-v1.5-onnx/model_int8.onnx

# 5) 补充实验 + 出图
python eval/run_extras_b.py  --split full --model-path models/bge-small-zh-v1.5-onnx/model_int8.onnx
python eval/run_quant_check.py --split full
python eval/run_backend_check.py --split full
python eval/plots.py --only e2,e3,e6
```

> 口径提醒：目录里同时存在 fp32 与 int8 时，`--encoder auto` 会**优先 fp32**（单路稠密高约 0.44 点）。
> 复现报告里的表必须显式钉 `--model-path …/model_int8.onnx`；每个结果 JSON 的 `config.model_file` 现在会写明到底用了哪个。
