# 交接单 · B（检索与融合）

> 依据 `执行顺序-接力棒与交接标准.md` 逐棒交付。下一位接手的人（A 集成 / C 重排 / D 评测 / E 汇报）只看本页。

## 0. B 在第几棒

**B 不是"一棒"，是四棒**（第 2、3、4 棒都要出手，第 6 棒是可选加分），第 7 棒还要给统稿供货。

上游进展（本地 README 同步到 11-21 版）：第 0 棒（接口冻结）✅、**第 1 棒（D，数据）✅**
（`data/build_data.py`、3507 段 / 3219 题 / 500 smoke，`doc_id` 从 `p_00001` 起、`dev.jsonl` 多一个只读 `source_id`），
**第 2 棒（A+B+C）← 现在在这里**。B 的代码已按 D 的新数据重跑一遍，E1/E2/E3 数字与换数据前逐位一致。

| 棒次 | 该棒要求 B 做什么 | 完成判据（原文） | 状态 |
| --- | --- | --- | --- |
| **第 2 棒**（A+B+C 合围） | BGE 编码全库 + FAISS 建索引 + topk 检索 | 给定一条 query 返回 top-5 doc_id，耗时 < 1 秒；整棒判据 `python eval/run_e1.py --split smoke` → Recall@1/5、MRR、F1、平均延迟 + 写 `results/raw/E1_dense_top5.json` | ✅ 检索半边全通（Recall@5 0.934、1.42 ms/查询、json 已落盘）；F1 记 null，因 C 的 reader 仍 STUB |
| **第 3 棒**（B、C 并行） | 手写 BM25、手写 RRF、接进主流程并加开关 `--retriever dense\|bm25\|hybrid`、`--alpha` | `python eval/run_exp.py --exp smoke` → 6 行全绿并写入 `results/raw/` | ✅ 4/6 绿（dense / bm25 / rrf α0.5 / rrf α1.0）；另 2 条是 rerank 配置，卡在 C 的 `Reranker`（`NotImplementedError`），如实 skip 并打印原因 |
| **第 4 棒**（全员，A 主持） | E2 七点 α、E3 三档切块 | 每个 run_id 有 json + CI + 能对应一张图；α 曲线形状可解释；重跑差 < 0.1 点 | ✅ 26 个 json、12 张 run 全带 bootstrap CI、两张图已出、重跑逐位一致 |
| **第 6 棒**（B，可选加分） | RM3 伪相关反馈 vs 最优配置 | 一张小表，提升或持平都要写 | ✅ 负结果（BM25 路 −1.74pt、RRF 路 −0.06pt），已写成结论 |
| **第 7 棒**（E 统稿，全员供稿） | 交第 5 章自己那段分析文字 + PPT 自己那几页 | 与主线一致 | ✅ `report/附录B_检索实验记录.md` + 5 页 PPT + 讲稿；第 2 章也已代写（原文归 E，E 可直接用或改写） |

---

## 1. 【第 2 棒交接】交棒人：B

```
交付物：retrieval/bge_encoder.py（BgeEncoder + FaissIndex）、retrieval/onnx_encoder.py、
        retrieval/bert_tokenizer.py、eval/run_e1.py::run_real()（补真跑，A 的原骨架保留）
验收方式：python eval/run_e1.py --split smoke --model-path models/bge-small-zh-v1.5-onnx/model_int8.onnx
        → Recall@1 0.884 / Recall@5 0.934 / MRR 0.907 / F1 待 C / 1.42 ms/query
        → results/raw/E1_dense_top5.json
已知限制：reader 仍是 STUB，F1/EM 记 null；把 C 的 ExtractiveReader 换掉后重跑同一条命令即可，无需改代码
下一棒是谁：A（拿这个 json 当 E1 分母）+ 第 3 棒（B 自己继续）
```

## 2. 【第 3 棒交接】交棒人：B

```
交付物：retrieval/bm25.py（手写 BM25：k1/b 公式带注释，倒排 + numpy 向量化 + 序列化）、
        retrieval/rrf.py（手写 RRF：score = Σ w/(k+rank)，k=60，weights=[1-α, α]）、
        retrieval/tokenization.py、retrieval/chunking.py、
        eval/run_exp.py::cmd_real()（只补真实运行分支，MATRIX/--list/--dry-run 未动）
验收方式：python eval/run_exp.py --exp smoke --split smoke   → [OK] 4/6 种配置跑完
        python eval/run_e3.py --retriever hybrid --split smoke --limit-queries 20   # 三档开关都在
已知限制：2 条 rerank 配置要等 C 的 Reranker（现在 raise NotImplementedError，脚本据此 skip 而不是崩）；
        开关值里 hybrid 与 rrf 同义（同一套 RRF，只是命名对齐接力棒原文）
下一棒是谁：第 4 棒的矩阵扫描（A 主持）
```

## 3. 【第 4 棒交接】交棒人：B

```
交付物：results/raw/E2_*.json（E2_bm25 / E2_dense / E2_rrf_alpha{0.0,0.2,0.4,0.5,0.6,0.8,1.0}）、
        E3_chunk_{para,256,512}.json、E3_chunk_{128,sentence,256o64,512o128}.json、
        results/tab/E2_alpha.csv、E3_chunk.csv、results/fig/E2_alpha_recall.png、E3_chunk.png
验收方式：python eval/run_e2.py --split full                               # ≈20 s（向量缓存命中）
        python eval/run_e3.py --split full --grid extended                 # ≈7 min
        python eval/plots.py --only e2,e3
已知限制：正式数字钉 int8（fp32 已补齐并做过量化对照，单路稠密差 0.44pt、融合后 R@5 相同）；
         绝对指标偏高属数据特性（CMRC 段落与问题同源），结论看相对差值与 CI
下一棒是谁：第 5 棒（C 的洞察实验拿 α=0.2 当固定检索配置）、第 7 棒（E 收图收表）
```

## 4. 【第 6 棒交接】交棒人：B（可选加分，已完成）

```
交付物：eval/run_e6_rm3.py、results/tab/E6_rm3.csv、results/raw/E6_{baseline,rm3}_*.json、
        results/fig/E6_rm3.png
验收方式：python eval/run_e6_rm3.py --split full
已知限制：只做了 γ=0.5 / fb_docs=10 / fb_terms=20 一组，未做参数细扫；负结果已写入报告附录 B.5
下一棒是谁：E（这一段是"局限与改进"页的素材：检索对了不等于改写有用）
```

---

## 5. 交付物总表

| 类别 | 路径 | 说明 |
| --- | --- | --- |
| 代码（冻结接口） | `retrieval/bm25.py` | 手写 BM25：倒排 + numpy 向量化 + 序列化 + 加权查询（供 RM3 用） |
| 代码（冻结接口） | `retrieval/bge_encoder.py` | `BgeEncoder`（onnx/st/hash 三后端 + 懒加载 + 查询前缀）+ `FaissIndex`（faiss/numpy 双后端 + 向量磁盘缓存） |
| 代码（冻结接口） | `retrieval/rrf.py` | `fuse` / `fuse_ids` / `alpha_weights` |
| 代码（新增） | `retrieval/tokenization.py` | 中文分词（jieba 优先，退化"单字+二元组"） |
| 代码（新增） | `retrieval/onnx_encoder.py`、`retrieval/bert_tokenizer.py` | 不依赖 torch/transformers 的 BGE ONNX 推理与 WordPiece 分词 |
| 代码（新增） | `retrieval/chunking.py` | 切块策略（整段/句级/定长 + 边界回退 + chunk→doc 压平） |
| 代码（新增） | `retrieval/hash_encoder.py` | 离线冒烟用的伪编码器（数字不可进报告） |
| 实验脚本 | `eval/run_e2.py`、`eval/run_e3.py`、`eval/run_e6_rm3.py`、`eval/run_extras_b.py`、`eval/run_quant_check.py`、`eval/run_backend_check.py` | E2 / E3 / E6 / 超参敏感度 / 量化 / 向量后端 |
| 实验入口（改 A 的文件） | `eval/run_exp.py` | **只补了 `cmd_real()` 真实运行分支**（接力棒第 3 棒）；`--list` / `--dry-run` / `MATRIX` 未动 |
| 实验入口（改 A 的文件） | `eval/run_e1.py` | **只补了 `run_real()`**（接力棒第 2 棒判据）；`--dry-run` 与骨架逻辑未动；reader STUB 时自动降级 |
| 辅助 | `eval/io_utils.py`、`eval/retrieval_eval.py` | 读写结果（严格按 INTERFACE 第 3 节）、指标包装（只调 `eval/metrics.py`） |
| 辅助 | `scripts/download_model.py`、`_dl.py`（多连接分段续传）、`make_ppt_b.py` / `inspect_ppt_b.py` / `check_text_fit.py`（PPT 生成与几何自检）、`md_to_docx_b.py`（md→宋体小四 docx）、`make_push_bundle.py`（打补丁包） | 下模型、生成 PPT/报告、打包 |
| 自检 | `tests/test_retrieval.py` | 13 项断言，`python -m pytest tests -q` |
| 结果 | `results/tab/E2_alpha.csv`、`E3_chunk.csv`、`E2_extras.csv`、`E2_quant.csv`、`E2_backend.csv`、`E6_rm3.csv` | 表格（进 git） |
| 结果 | `results/raw/E2_*.json`、`E3_*.json`、`E6_*.json`、`E1_dense_top5.json` | 逐题明细（被 .gitignore 挡住，需本地复跑） |
| 图 | `results/fig/E2_alpha_recall.png`、`E2_latency.png`、`E3_chunk.png`、`E6_rm3.png` | 报告与 PPT 用（进 git） |
| 文档 | `report/第2章_相关工作.md`、`report/附录B_检索实验记录.md` | 报告正文两章 |
| 文档 | `docs/B_检索模块说明.md`、`docs/B_交接单.md`、`docs/pitfalls.md`（补 B 条目） | 说明书 / 交接单 / 踩坑 |
| 演示 | `ppt/检索部分_B.pptx`、`ppt/讲稿_B.md` | 5 页片段 + 逐页讲稿 |

## 6. 一键验收

```powershell
$env:PYTHONPATH="$PWD\.pylibs;$PWD"      # .pylibs 是本地 --target 装的依赖
$env:MPLCONFIGDIR="$PWD\.mplcache"

python eval/check_data.py --strict                            # 数据四项断言全过
python -m pytest tests -q                                     # 13 passed
python eval/run_e1.py --split smoke --model-path models/bge-small-zh-v1.5-onnx/model_int8.onnx   # 第 2 棒判据
python eval/run_exp.py --exp smoke --split smoke              # 第 3 棒判据（4/6，2 条重排属 C）
python eval/run_e2.py --split full                            # ≈20 s（编码命中缓存）
python eval/run_e3.py --split full --grid extended            # ≈7 min
python eval/run_e6_rm3.py --split full                        # 加分项
python eval/plots.py                                          # 出 4 张图
```

**验收线**（来自 `执行顺序-接力棒与交接标准.md`）：
- 单查询 < 1 s → 实测 1.8–3.5 ms；
- 同配置重跑两遍差值 < 0.1 个点 → 固定 seed + 稳定排序，重跑逐位一致；
- `recall@5` 落在 0.85–0.96 → 实测 0.934（E1 smoke）/0.949–0.985（full 各配置），**full 略高于上限**，原因见已知限制第 1 条。

## 7. 已知限制

1. **绝对指标偏高**：CMRC2018 的段落与问题同源（同一篇维基文章），词面重合度高，
   纯 BM25 的 R@5 就有 0.976。**结论请看相对差值与置信区间，不要看绝对值。**
2. **量化口径**：正式数字用 `model_int8.onnx`；fp32 已补齐并做过对照（`results/tab/E2_quant.csv`）——
   单路稠密 fp32 0.9531 / int8 0.9487，**融合后 R@5 都是 0.9854**。E3 七档未在 fp32 下重跑。
   两个权重都在目录里时 `--encoder auto` 会优先 fp32，复现请显式 `--model-path .../model_int8.onnx`。
3. **向量后端**：装了 `faiss-cpu` 1.16.0，两后端都跑过（`results/tab/E2_backend.csv`）：
   top-100 **集合 100% 一致、逐位 99.75% 相同**，端到端指标逐位一致；预热后 faiss 0.201 ms /
   numpy 0.268 ms（差 0.07 ms，可忽略）。**正式数字由 faiss 跑出**（各 run 的 config.index_backend 有记录），
   换机器装不上 faiss 会自动退 numpy，指标不变。
4. **jieba 路径**：`.pylibs` 里是手工解包的 sdist，不是 wheel；换机器请 `pip install jieba`，
   或直接用 ngram 兜底（结论会变，见 `docs/pitfalls.md`）。
5. **E6 RM3 是加分项**：已实现并跑通，未做参数细扫。

## 8. 下一棒具体怎么接

- **A（集成）**：`results/raw/E1_dense_top5.json` 直接当 E1 分母；`run_e1.py::run_real()` 已接好检索侧，
  等 C 的 reader 落地后同一条命令就能补齐 F1/EM。所有 run 的 json 严格按 INTERFACE 第 3 节，汇总脚本可直接消费。
- **C（重排 + 阅读器）**：`E2_rrf_alpha0.2.json` 的 `per_question[].ranked`（top-10 段落 id）
  可直接当重排候选集合，不必重跑检索。你的 `Reranker` 一落地，`run_exp.py --exp smoke` 的 6/6 就齐了。
- **D（评测）**：所有指标都经由 `eval/metrics.py` 算，B 的脚本只是包装；CI 用 `bootstrap_ci`（1000 次，seed=42）。
- **E（汇报）**：`ppt/检索部分_B.pptx` 是 5 页独立片段，按「检索原理 → α 曲线 → 切块曲线」
  插到主 PPT 第 4–8 页区间；讲稿在 `ppt/讲稿_B.md`，含"可能被问到的问题"对照表。

## 9. 越界声明（碰了别人负责的文件，请对应的人过一眼）

自查时逐条比对《接力棒》，有 3 处动了别人名下的文件。**都不是偷偷改的，都在补丁包里单列**，
理由与回退方式见 `docs/B_自查报告.md` 第 2 节。摘要：

| 文件 | 归属 | 我改了什么 | 为什么 | 怎么回退 |
| --- | --- | --- | --- | --- |
| `eval/run_e1.py` | A（第 2 棒"写 run_e1.py"） | 只填 `run_real()` + 4 个 CLI 参数 | 第 2 棒判据 `run_e1.py --split smoke` 要能跑 | `git checkout -- eval/run_e1.py`，检索仍可经 `run_e2.py` 调用 |
| `eval/run_exp.py` | A（骨架） | 只补 `cmd_real()` + 3 个参数 | 第 3 棒原文点名 B"接进主流程、加 `--retriever`/`--alpha`" | `git checkout -- eval/run_exp.py` |
| `eval/plots.py`、`results/fig/*.png` | D（第 4 棒"出版式统一的所有图"） | 写了 4 张图 | README 把 E2/E3 图列在 B 名下 | D 可直接覆盖重出，B 不占坑 |

另有两处共享文件：`requirements.txt`（只在末尾追加，未删原行）、`docs/pitfalls.md`（填了标着"（B 补）"的条目，
并把误占的 D 待记项标注为"原 D 待记项的实测答案"）。

**标准没要求、但我加了的东西**（可整段丢弃，不影响验收）：`tests/test_retrieval.py`（13 项自检）、
三个补充实验（超参 / 量化 / 后端）、`scripts/md_to_docx_b.py`（md→docx）、PPT 生成与几何自检脚本、
`scripts/make_push_bundle.py`（打补丁包）、`report/第2章_相关工作.md`（方案把第 2 章分给 B）。

