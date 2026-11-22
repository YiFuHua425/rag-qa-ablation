# C 的模块说明：交叉编码器重排 + 抽取式阅读器 + 位置控制

> 对应 `执行顺序-接力棒与交接标准.md` 第 3、4、5 棒（C）。
> 只讲「怎么实现的、为什么这么实现」，实验数字见 `docs/C_交接单.md`。

| 文件 | 状态 | 一句话 |
|---|---|---|
| `rerank/cross_encoder.py` | READY | `Reranker`：bge-reranker-base，(query, passage) 成对打分，只对前 depth 个候选重排 |
| `reader/context_builder.py` | READY | `build_context`：四档顺序 + 整段截断 + 返回 `used_doc_ids` 与诊断信息 |
| `reader/extractive_qa.py` | READY | `ExtractiveReader`：滑窗 + offset 还原 + batch，输出 `(answer, logit)` |
| `eval/c_pipeline.py` | 新增 | E4/E5 共用的「候选 → 重排 → 上下文 → 阅读器」链路与候选缓存 |
| `eval/run_e4.py` | 新增 | E4：`{off,d20,d50,d100}` × 精度/延迟 |
| `eval/run_e5.py` | 新增 | E5：`order ∈ {natural, first, middle, last}` × `max_chars ∈ {2000,4000}` |
| `eval/plots_c.py` | 新增 | `E4_pareto.png`、`E5_position.png` |
| `scripts/download_models_c.py` | 新增 | 两个权重的下载（modelscope / hf-mirror 双源） |
| `tests/test_c_modules.py` | 新增 | 11 项断言，不需要权重 |

---

## 1. 交叉编码器重排（`Reranker`）

### 1.1 为什么是「重排」而不是「重检索」

双塔（双编码器）把 query 和 passage 分别编码成向量，内积近似相关度——**query 和 passage
在编码阶段互不可见**，词级交互全靠一个 512 维向量兜着。交叉编码器把 `[query, passage]`
拼成一条序列喂进 Transformer，每个 query token 都能和 passage token 直接做注意力，
精度明显更高，代价是**不能在离线阶段预计算**：每个 (query, passage) 都要过一遍模型。

所以工程上的标准做法是「双塔召回 + 交叉编码器精排」：
先用 RRF 融合（B 的模块）拿 100 个候选，再让交叉编码器对其中前 `depth` 个重新打分。
`depth` 就是 E4 的自变量：**20 / 50 / 100 三档，精度涨多少、延迟涨多少**。

### 1.2 打分与排序

```
s(q, d) = f_θ([CLS] q [SEP] d [SEP])          # 单 logit 的回归头
ranked  = sort(s(q, d_i), desc)[:topn]
```

* **输出是 logit，不是概率**：`bge-reranker-base` 的 `num_labels=1`，
  直接取 `logits[:, 0]`；若换成二分类头（`num_labels=2`）则取 `logit_1 - logit_0`。
* **只用于排序**：logit 与 BM25 分、余弦相似度都不同量纲，**不允许相加**。
  报告里凡是「融合分数」一律指 RRF，重排分数只出现在排序结果里。
* **候选不丢**：`depth` 之外的候选（第 depth+1 … 100 名）**原序追加在返回列表尾部**。
  原因是 E4 的 Recall@5 在重排后的前 5 名上算——如果重排顺手把候选删了，
  Recall 的下降就分不清是"重排排错了"还是"候选被丢了"。
* **文本从哪来**：冻结的 `rerank(query, candidates, topn)` 只给 `(doc_id, score)`，
  段落原文得靠 `set_doc_map()` / `set_corpus(docs)` 预先灌进去。
  查不到原文的候选记 `-inf` 排到最后，不抛异常（宁可退化，也不让一条脏数据炸掉整批实验）。

### 1.3 三个后端

| backend | 依赖 | 用途 |
|---|---|---|
| `hf` | torch + transformers | **正式口径**，加载 `models/bge-reranker-base` |
| `onnx` | onnxruntime + tokenizer | 没有 torch 的机器（本仓库的 `onnx/model.onnx` 也可直接用） |
| `lexical` | 无 | **离线占位**：字符二元组重合度，只为验证链路，数字不可进报告 |

`lexical` 会在第一次调用时抛 `RuntimeWarning`，且结果 json 的 `config.rerank_backend`
会记成 `lexical`——写报告时一眼能看出这批数字不能引用。

### 1.4 延迟口径（E4 的横轴）

* **报出来的**：`benchmark_latency()`，`batch=1`、`torch.set_num_threads(1)`、重复 3 次取中位数，
  单位 ms/query。这是 config.yaml `latency` 段规定的口径，也是 Pareto 图的横轴。
* **算指标用的**：`batch_size=16` + 多线程（脚本里 `score_threads = cpu_count - 2`）。
  两件事分开跑，否则「多线程吞吐」会被误读成「单查询延迟」。
* 端到端总延迟 = 检索（BM25+稠密+RRF） + 重排 + 阅读器，三段各自测、分别记进 json，
  报告里可以拆开看瓶颈在哪。

---

## 2. 上下文拼装（`build_context`）

签名（冻结）：

```python
build_context(doc_ids, doc_map, order="natural", max_chars=2000,
              gold_doc_id=None) -> (context_text, used_doc_ids)
```

### 2.1 四档顺序

| order | 行为 | 用途 |
|---|---|---|
| `natural` | 保持检索/重排顺序 | 正常实验 |
| `gold_first` | 金标挪到第 1 段 | E5 |
| `gold_middle` | 金标挪到正中间（5 段 → 第 3 段） | E5 |
| `gold_last` | 金标挪到最后一段 | E5 |

金标不在候选里（检索没召回）时**不做任何移动**——E5 会把 `gold_in_candidates=0`
那批题单独统计，避免"检索没召回"被算成"位置效应"。

### 2.2 截断：整段拼入，超限即停

不截半个段落，是因为半个段落会让 F1 的差异混入「句子被切断」这个混淆因素。
第一段如果单独就超 `max_chars` 也照拼（否则上下文为空，阅读器必然空答案），
溢出量记在 `overflow_chars` 里。—— CMRC 段落最长 980 字，`max_chars=2000` 时不会触发。

### 2.3 诊断信息旁路

返回值必须是二元组（别处按两元解包），所以每次调用后把本轮统计写进模块级
`LAST_STATS`，用 `get_last_stats()` 读：

```json
{"order": "gold_last", "gold_in_candidates": true, "gold_used": false,
 "gold_position": null, "n_candidates": 5, "n_used": 3, "chars": 1505,
 "max_chars": 2000, "dropped_doc_ids": ["d", "e"], "missing_doc_ids": []}
```

E5 靠 `gold_used` 把失败拆成两类：**金标没进上下文**（截断/排序问题）与
**金标进去了但阅读器没答对**（模型问题）。这两种失败的改进方向完全不同，
混在一起报一个 F1 是没法给结论的。

---

## 3. 抽取式阅读器（`ExtractiveReader`）

### 3.1 为什么是抽取式

* 确定性、可复现：同一输入永远同一输出，消融实验的差异只来自我们改的那个旋钮；
* 参数量小（roberta-base 约 1.1 亿），CPU 也跑得动，不用和显存/量化搏斗；
* 生成式（Qwen2.5-7B）只做视频里的定性演示，**不承担主结论**——
  生成式有幻觉，F1 的差异会混入"编造答案"这种与检索无关的因素。

### 3.2 前向与解码

```
context 超长 → truncation="only_second" + doc_stride=128 + max_length=512 → 多个 feature
每个 feature: start_logits / end_logits (各 512 维)
取 top-n_best 个 start，对每个 start 在 [start, start+max_answer_len) 内取 argmax end
score = start_logit + end_logit，只保留 sequence_ids == 1（context 侧）的 token
跨 feature 取最高分 → offset_mapping 还原成原文字符区间
```

几个坑（都在代码注释里）：

1. **不过滤 `sequence_ids`**：模型有时把问题里的词当成答案，F1 会虚高。
2. **不滑窗**：CMRC 段落均值 509 字 ≈ 700 token，超过 512 就直接截掉后半段，
   一半的题会"答案不在窗口里"，F1 无解释地掉——本仓库用 `doc_stride=128` 滑窗。
3. **返回原文而不是 token 串**：中文 BERT 是字级分词，拼 token 会插入空格，
   F1 会莫名掉几个点。
4. **空答案**：还原出空串 → `("", 0.0)`；`null_score` 默认 -100（等于不启用），
   换 SQuAD2 风格权重时才需要调。

### 3.3 批量

`answer_batch(pairs)` 把**所有题的 feature 汇总**后按 `batch_size` 前向，
比逐题调用快 5–10 倍。E4 有 4 个 arm、E5 有 8 个 arm，同一条上下文要反复读，
不走批量的话光阅读器就得跑一晚上。

---

## 4. 复现命令

```bash
# 权重（约 1.5 GB，模型不进 git）
python scripts/download_models_c.py --source modelscope --only rerank
python scripts/download_models_c.py --source hf --only reader --parts 12
python scripts/download_model.py --parts 8          # BGE 编码器 ONNX（检索用）

# 自检（不需要权重）
python -m pytest tests/test_c_modules.py -q

# E4：重排与候选深度
python eval/run_e4.py --split smoke --limit-queries 200

# E5：位置偏差
python eval/run_e5.py --split smoke --limit-queries 300

# 只画图
python eval/plots_c.py
```

Windows 上建议激活 `.venv` 后跑，`MPLCONFIGDIR` 由 `plots.py` 自动指到仓库内（避免沙箱写不进去）。
