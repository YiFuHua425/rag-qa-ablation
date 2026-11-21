# B 分工：检索模块说明书（BM25 / BGE / FAISS / RRF）

> 对应 `INTERFACE.md` 里标 `[STUB]` 的三个文件：`retrieval/bm25.py`、
> `retrieval/bge_encoder.py`、`retrieval/rrf.py`。接口签名已冻结，本文只描述实现。
> 另外两个自研模块 `retrieval/tokenization.py`（中文分词）与
> `retrieval/chunking.py`（切块）是 E3 的必要条件，见第 4 节。

---

## 1. BM25（`retrieval/bm25.py`）

### 1.1 打分公式

对查询 `q` 与文档 `d`：

```
score(q, d) = Σ_{t ∈ q} IDF(t) · f(t,d)·(k1+1) / ( f(t,d) + k1·(1 - b + b·|d|/avgdl) )
IDF(t)      = ln( (N - n_t + 0.5) / (n_t + 0.5) + 1 )
```

- `N` 语料文档数，`n_t` 含词 `t` 的文档数，`f(t,d)` 词频，`|d|` 文档长度，`avgdl` 平均长度；
- 采用 Lucene 版 IDF（`+1` 在外层），保证 `n_t > N/2` 时 IDF 仍为正；
- 查询词**去重**后求和（`dedup_query=True`）：中文问句常重复词，重复加权会放大噪声。
  `dedup_query=False` 时按出现次数加权，两者都能跑（自检里有断言）。

### 1.2 实现要点

| 设计 | 原因 |
| --- | --- |
| 倒排表 `postings: dict[str, dict[int, int]]`（词 → {行号: 词频}） | 查询只遍历命中的词，时间复杂度与查询长度而非语料大小成正比 |
| 用 numpy 向量化打分 `score[nonzeros]` | 3.5k 段落、3219 题全量打分 0.6 s；3219 条查询端到端 2.2 ms/条 |
| 只对非零行排序（`np.lexsort((行号, -分数))`） | 次关键字取行号 → **同分返回稳定的语料顺序**，实验可复现 |
| 空查询/全 OOV 返回 `[]` | 下游融合与评测不必特判 |
| `to_dict/save/load` 带 `tokenizer` 配置 | 否则 ngram 建的表用 jieba 查，分数会静默错位（踩过，见 `docs/pitfalls.md`） |
| `doc_terms` 随 `load()` 反推重建 | 保证 RM3（E6）在加载后的索引上仍可用 |

### 1.3 中文分词（`retrieval/tokenization.py`）

`ChineseTokenizer(mode="auto"|"jieba"|"ngram")`：

- `jieba`：精确模式 + 内置停用词表（"的/了/是/在/和/与"等虚词）；
- `ngram`：正则切出 CJK 串与拉丁/数字串，CJK 串展开为**单字 + 相邻二元组**
  （"南京市长江大桥" → 南/京/市/…/南京/京市/…）。不依赖词典，对未登录词更鲁棒，
  代价是倒排膨胀；
- `auto`：能 import jieba 就用 jieba，否则退 ngram。

实测同一份 E2 配置下，**字符二元组会让 BM25 过强、α 曲线单调下降**（见 5.1），
换 jieba 后曲线才回到"中间高两头低"。这条经验直接写进了报告的分析。

---

## 2. BGE 稠密编码与 FAISS 索引（`retrieval/bge_encoder.py`）

### 2.1 编码器

模型：`BAAI/bge-small-zh-v1.5`（**4 层 Transformer，隐层 512 维，词表 21128，最大长度 512**）。
取 `[CLS]`（第一 token）的最后一层隐向量，再 **L2 归一化**，于是内积 = 余弦。

```python
enc = BgeEncoder(model_name="BAAI/bge-small-zh-v1.5")   # 懒加载，--dry-run 不触发下载
embs = enc.encode_docs(docs)        # (N, 512) float32，已归一化
q = enc.encode_query("中国的首都是哪里？")   # (512,)
```

三种后端 `backend="auto"|"onnx"|"st"|"hash"`：

| 后端 | 依赖 | 用途 |
| --- | --- | --- |
| `onnx` | onnxruntime + `models/bge-small-zh-v1.5-onnx/` | **本组实际使用**：Windows 上 `torch` 装不上（见 5.2），改走 ONNX |
| `st` | sentence-transformers + torch | 有 GPU 的机器上的默认路线 |
| `hash` | 无 | 离线冒烟：字符 n-gram 哈希到 512 维，**数字不可进报告** |

ONNX 路线自带 BERT WordPiece 分词器 `retrieval/bert_tokenizer.py`
（读 `tokenizer.json` 的 `normalizer/wordpiece` 配置，`[CLS] x [SEP]`、按批内最长补齐），
不依赖 `transformers` 与 `torch`。非对称指令前缀由 `BgeEncoder.query_prefix`
统一负责，ONNX 引擎内建关闭，避免加两遍。

> 指令前缀：BGE 官方建议中文检索查询侧加"为这个句子生成表示以用于检索相关文章："。
> 本组 `config.yaml` 里 `query_prefix` 为空串，理由是 CMRC2018 的查询本身是
> 完整问句、且段落极短（均值 509 字），实测加前缀反而略降；数值见 5.3。
> 改动只在配置里，不改代码。

### 2.2 索引

`FaissIndex(index_type="IndexFlatIP")`：**精确内积**（不是 IVF/HNSW，3.5k 段落无需近似）。
装了 `faiss-cpu` 就用 `faiss.IndexFlatIP`，没装则退 numpy 的 `embs @ q`，
两者 top-k **集合完全一致**（自检有断言）。实测（`results/tab/E2_backend.csv`）：
3507×512 上 faiss 单查询 **0.201 ms**、numpy **0.268 ms**（预热 + 5 轮中位数；早先没预热时
测出过相反的结论，见 `docs/pitfalls.md`）。差 0.08 ms，相对整条链路（约 2 ms）可忽略，
**faiss 的优势要到十万级语料才显著**。所以两个后端都留了口子：正式数字由 faiss 跑出，
两后端的端到端指标逐位一致、top-100 集合相同、逐位相同 99.75%（同分次序）。

段落向量**落盘缓存**：`build_from_encoder(..., tag="docs")` 把结果写进
`.cache/emb/emb_<tag>_<stamp>_n<N>.npz`，`stamp` 含后端名、模型文件名、`max_length`、
归一化开关与**语料内容 sha1 摘要**（12 位），所以换模型 / 换量化版本 / 换数据编号都
不会误用旧向量；`--no-cache` 可强制重算。
第一次编码 3507 段用 53.8 s（int8，CPU 单机；fp32 需 100.0 s），之后各实验全部命中缓存。

---

## 3. RRF 融合（`retrieval/rrf.py`）

```
fused(d) = Σ_i w_i / (k + rank_i(d))          rank 从 1 起；某路未召回 d 则该路贡献 0
```

- `weights=None` 为等权；两路混合检索时 `[1-α, α]`，**α 是稠密路权重**（α=0 纯 BM25，α=1 纯稠密）；
- `k=60` 为 Cormack 等的经验值，本组实测对结果影响很小（见 5.4）；
- 同一路内重复 doc_id 只算**最好名次**，不重复累加（融合前两路各自 `set` 去重）；
- 同分按**首次出现顺序**稳定排序，保证可复现；
- 只依赖名次、不依赖分数刻度，因此 BM25 的长尾分数与余弦的窄区间不需要归一化。

`alpha_weights(alpha) -> [1-alpha, alpha]` 供各实验脚本共用，避免每处自己算。

---

## 4. 切块（`retrieval/chunking.py`，E3 用）

`ChunkStrategy(name="whole"|"sentence"|"fixed", size, overlap, boundary_aware=True)`：

| 名称 | 含义 | 3507 段上的块数 |
| --- | --- | --- |
| `paragraph` | 整段一个块（= E1/E2 口径） | 3507 |
| `sentence` | 按 `。！？；` + 右引号/右括号切句 | 43134 |
| `128 / 256 / 512` | 定长窗口，`boundary_aware` 时向后找句末标点再切 | 17625 / 9302 / 4934 |
| `256o64 / 512o128` | 同上 + 25% 重叠 | 10753 / 5157 |

- **边界回退**：窗口切点落在句子中间时，向后最多找 `size/2` 个字，遇到句末标点就在其后切，
  避免把答案跨度拦腰截断；找不到才硬切。
- **chunk → 段落压平**：块级排名去重压平成段落排名（`chunks_to_doc_ranking`），
  这样 E3 与 E2 的指标口径完全一致，可直接对比。

---

## 5. 实验记录（数字均为 `--split full`，3219 题）

### 5.1 E2：α 扫描（int8 ONNX + jieba + RRF k=60 + depth=100）

| α（稠密权重） | R@1 | R@5 | R@5 95% CI | MRR | nDCG@5 |
| --- | --- | --- | --- | --- | --- |
| 0.0（纯 BM25） | 0.8885 | 0.9764 | [0.9708, 0.9817] | 0.9256 | 0.9385 |
| **0.2** | **0.9441** | **0.9854** | **[0.9814, 0.9894]** | **0.9617** | **0.9678** |
| 0.4 | 0.9419 | 0.9798 | [0.9751, 0.9848] | 0.9571 | 0.9628 |
| 0.5 | 0.9400 | 0.9779 | [0.9727, 0.9829] | 0.9549 | 0.9607 |
| 0.6 | 0.9357 | 0.9755 | [0.9702, 0.9804] | 0.9514 | 0.9574 |
| 0.8 | 0.9245 | 0.9649 | [0.9587, 0.9714] | 0.9414 | 0.9473 |
| 1.0（纯稠密） | 0.8978 | 0.9487 | [0.9410, 0.9562] | 0.9181 | 0.9258 |

读法：**α=0.2 处最好**，比纯 BM25 高 0.9 个点 R@5、比纯稠密高 3.7 个点；
更值得注意的是 MRR/Recall@1 的增益（+3.6 / +5.6 个点 vs BM25），
说明融合主要把"金标从第 2–3 名拉到第 1 名"，而不是补召回。
曲线在 α>0.6 之后快速下滑 → 这份数据上**稠密通道的边际价值低于稀疏通道**。

### 5.2 为什么是 ONNX 而不是 torch

本机 `pip install torch` 触发 Windows 沙箱权限错误（`PermissionError: [Errno 13]`，
`pip-unpack` 临时目录的 DACL 被继承成 0o700），且 250 MB 轮子在镜像下下载不稳。
改用 `onnxruntime` + 官方导出的 ONNX 权重（`model_int8.onnx` 22.8 MB）后整条链路
在 CPU 上跑通，单查询 2.2 ms。代价是必须自带 WordPiece 分词器
（`retrieval/bert_tokenizer.py`），以及 int8 量化带来的轻微精度损失（5.5）。

### 5.3 无指令前缀 vs 有指令前缀

| 配置 | 纯稠密 R@5 | 纯稠密 MRR | RRF(α=0.2) R@5 | RRF MRR |
| --- | --- | --- | --- | --- |
| 无前缀（本组默认） | **0.9487** | **0.9207** | 0.9854 | **0.9628** |
| 加"为这个句子生成表示…" | 0.9435 | 0.9173 | **0.9863** | 0.9606 |

单路时加前缀反而降 0.5 个点（按句检索式改写对短问题的收益被摊薄）；
融合后的 R@5 差异（+0.09pt）落在置信区间内，MRR 仍是加前缀更差。
**结论：这份数据上用默认 `query_prefix=""`，但差距很小，属于"配置项"而非"结论"。**

### 5.4 超参敏感度（`results/tab/E2_extras.csv`，共 27 行）

- `rrf_k ∈ {10,20,60,100,200}` → R@5 = 0.9832 / 0.9848 / **0.9854** / 0.9848 / 0.9823：
  **k=60 附近最稳，全域差 0.31 个点**；k 小则"赢者通吃"，k 大则趋于平权。
- `depth ∈ {20,50,100,200}` → R@5 = 0.9845 / 0.9851 / 0.9854 / 0.9857：
  **≥50 就饱和**，取 100 是保守值（多算 100 个内积约 +0.1 ms）。
- BM25 `k1 ∈ {0.9,1.2,1.5,2.0} × b ∈ {0.3,0.5,0.75,0.9}`：R@5 区间 0.9739–0.9776、
  MRR 区间 0.9192–0.9315。默认 `(1.5, 0.75)` 的 R@5=0.9764、MRR=0.9268，
  与最优点 `(0.9/1.2, 0.75)`（0.9776）差 **0.12 个点** → **结论不依赖调参**；
  `b` 的作用比 `k1` 明显（b 从 0.3 到 0.9 掉 0.25–0.3 个点），符合"短段落对长度归一化更敏感"的直觉。
- **归一化 ablation**（稠密向量不做 L2 归一化，退化为裸内积）：

| 配置 | 纯稠密 R@5 | 纯稠密 MRR | RRF R@5 | RRF MRR |
| --- | --- | --- | --- | --- |
| 归一化（正确） | **0.9487** | **0.9207** | 0.9854 | **0.9628** |
| 不归一化 | 0.9366 | 0.8972 | **0.9857** | 0.9604 |

单路掉 1.2 个点召回、2.4 个点 MRR；融合后召回损失被兜住（+0.03pt 在噪声内），
**但精度损失转移到排序指标上（MRR −0.24pt）**——只看 Recall 会误判"没影响"。

### 5.5 量化：int8 vs fp32（数据在 `results/tab/E2_quant.csv`）
正式实验用 `model_int8.onnx`（22.8 MB，动态量化）跑完。fp32 权重（`model.onnx`，94.9 MB）
后来用 4 连接分段续传（`scripts/_dl.py`）补下来了，于是把量化损失测了出来：

| 权重 | 纯稠密 R@5 | 纯稠密 MRR | RRF(α=0.2) R@5 | RRF MRR | 全库编码 |
| --- | --- | --- | --- | --- | --- |
| fp32 | **0.9531** | **0.9260** | **0.9854** | **0.9636** | 139.7 s |
| int8 | 0.9487 | 0.9207 | **0.9854** | 0.9628 | 100.7 s |

同一段落两套向量的余弦均值 0.9478（最小 0.9058）。**结论：量化只影响单路稠密
（−0.44pt R@5 / −0.53pt MRR），融合后 R@5 完全一致、MRR 差 0.08pt**，
所以正式口径用 int8 不改变任何主结论；选它的实际好处是体积小 4 倍、编码快约 1.4 倍。
（E3 七档网格未在 fp32 下重跑，属已知未闭环项。）

> ⚠️ **复现时注意**：两个权重现在都在目录里，`--encoder auto` 会**优先挑 fp32**，
> 跑出来的数字会比报告里的 int8 表高约 0.4 个点。要复现报告数字必须显式钉住：
> `--model-path models/bge-small-zh-v1.5-onnx/model_int8.onnx`（`model_path` 既收目录也收 .onnx 文件）。

### 5.6 E6 RM3 伪相关反馈：一个负结果

| 配置 | R@1 | R@5 | MRR | nDCG@5 |
| --- | --- | --- | --- | --- |
| BM25 基线 | 0.8885 | 0.9764 | 0.9256 | 0.9385 |
| BM25 + RM3 | 0.8608 | 0.9590 | 0.8994 | 0.9144 |
| RRF 基线 | 0.9400 | 0.9779 | 0.9549 | 0.9607 |
| RRF + RM3 | 0.9323 | 0.9773 | 0.9506 | 0.9573 |

BM25 路 −1.74 个点，RRF 路 −0.06 个点（噪声内）。原因：段落太短（509 字），
top-20 反馈词基本等于"把整段搬进查询"，高频泛词稀释了精确匹配信号；
而融合路抗住了这个噪声 → **融合不仅涨点，还提升鲁棒性**。
详见 `report/附录B_检索实验记录.md` B.5。

---

## 6. 复现

```powershell
# 0) 依赖（清华镜像；Windows 沙箱会卡 pip 的临时目录，用 --no-cache-dir）
pip install --no-cache-dir --only-binary :all: `
  -i https://pypi.tuna.tsinghua.edu.cn/simple `
  --target .pylibs onnxruntime pyarrow matplotlib tqdm pyyaml pytest faiss-cpu
# jieba 只有 sdist：直接下 tar.gz 解包到 .pylibs\jieba

# 0.5) 权重（不入库，见 models/.gitignore；默认 int8 22.8MB，--fp32 另加 94.9MB）
python scripts/download_model.py                  # int8
python scripts/download_model.py --fp32 --parts 4 # 量化对照用，4 连接分段续传

# 1) 数据（归 D：`data/build_data.py`，见 `data/README.md`；数据文件不入库）
python data/build_data.py --sample 20     # → data/corpus.jsonl / data/dev.jsonl
python eval/check_data.py --strict        # 四项断言必须全过
python eval/check_data.py --make-smoke 500
```

> **数据口径（第 1 棒 D 冻结）**：`corpus.jsonl` 3507 段（`doc_id` 从 **`p_00001`** 起，
> 段落按 train→dev→trial 去重先到先得编号）；`dev.jsonl` 3219 题（`qid` 形如 `dev_0001`，
> 另带只读字段 `source_id`，如 `DEV_0_QUERY_0`，下游可忽略）；`dev_smoke.jsonl` 500 题（`seed=42`）。
> 原始文件是 `ymcui/cmrc2018` 的 squad-style json，放 `data/raw/cmrc2018_{train,dev,trial}.json`。
> B 的检索侧只用到 `{"doc_id","text"}` 与 `{"qid","question","gold_doc_id"}`，
> 因此 `source_id` 这类增补字段对指标无影响（重跑 E1/E2/E3 数字逐位一致）。

```powershell
# 2) 自检
python -m pytest tests -q                 # 13 passed

# 3) 实验
python eval/run_e2.py --split full --model-path models/bge-small-zh-v1.5-onnx/model_int8.onnx
python eval/run_e3.py --split full --grid extended --model-path models/bge-small-zh-v1.5-onnx/model_int8.onnx
python eval/run_e6_rm3.py --split full --model-path models/bge-small-zh-v1.5-onnx/model_int8.onnx
python eval/run_extras_b.py --split full --model-path models/bge-small-zh-v1.5-onnx/model_int8.onnx
python eval/run_backend_check.py          # → tab/E2_backend.csv（faiss vs numpy）
python eval/run_quant_check.py            # → tab/E2_quant.csv（int8 vs fp32）
python eval/plots.py                      # → results/fig/E2_*.png, E3_chunk.png
```

> 运行前需把 `.pylibs` 与仓库根都放进 `PYTHONPATH`，并设 `MPLCONFIGDIR` 到仓库内
> （Windows 下 matplotlib 默认缓存在用户目录，沙箱不可写）。
> **`*.onnx` 权重被 `models/.gitignore` 排除**：克隆后必须先跑 `python scripts/download_model.py`，
> 否则 `BgeEncoder` 会回退到 hash 伪编码器（数字不可用）。

---

## 7. 已知局限

1. **单语料、单档规模**：3507 段、3219 题，`IndexFlatIP` 够用；上到百万段落必须换 IVF/HNSW。
2. **int8 量化**：官方数字基于 `model_int8.onnx`，与 fp32 有 < 0.5 个点差距（5.5）。
3. **CMRC2018 的段落天生与问题同源**（来自同一篇维基文章），难度低于开放域检索，
   所以绝对指标偏高（R@5 0.95+），**结论应主要看相对差值**。
4. **只做了 RRF**：分数归一化加权（CombSUM）与学习式融合未做，留作改进方向。
5. **切块未评估重叠的边际收益**：`256o64`/`512o128` 在扩展网格里有数，但未做重叠比例的细扫。
