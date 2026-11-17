# 接口契约（第 0 棒冻结版）

> **改动本文件必须全员同意。** 签名一旦变了，五个人的代码同时失效。
> 状态标记：`[STUB]` = 空实现，待负责人替换；`[READY]` = 可直接使用。

---

## 1. 数据格式

### `data/corpus.jsonl`（一行一篇文档，UTF-8）

```json
{"doc_id": "p_00312", "text": "段落原文……"}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `doc_id` | str | 全局唯一，**不许重复**；建议 `p_` + 5 位序号 |
| `text` | str | 段落原文，保留原标点，不做分词 |

### `data/dev.jsonl`（一行一个问题）

```json
{"qid": "dev_0001", "question": "……？", "gold_doc_id": "p_00312",
 "answer_text": "标准答案片段", "answer_start": 57}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `qid` | str | 全局唯一 |
| `question` | str | 问题原文 |
| `gold_doc_id` | str | **必须存在于 corpus** —— 这是检索的 ground truth |
| `answer_text` | str | 标准答案，**必须是 `gold_doc_id` 段落的子串** |
| `answer_start` | int | 答案在段落中的起始字符偏移（Python 字符串下标） |

> **硬约束（`check_data.py` 会断言）**：`answer_text` 越界数必须为 **0**。
> 答案不在金标段落里，后面所有指标都在算错东西。

### 评测两档

| split | 规模 | 用途 |
|---|---|---|
| `smoke` | 随机 500 题（固定 seed=42） | 开发迭代，跑得快 |
| `full` | dev 全集 | 终稿数字，写进报告 |

---

## 2. 模块签名

### `retrieval/bm25.py` — B 负责 `[STUB]`

```python
class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75): ...
    def fit(self, docs: list[dict]) -> None: ...          # [{"doc_id","text"}]
    def search(self, query: str, topk: int) -> list[tuple[str, float]]: ...
```

**实现要点（写进报告第 4 章）**：
`score(q,d) = Σ_{t∈q} IDF(t) · f(t,d)(k1+1) / ( f(t,d) + k1·(1 − b + b·|d|/avgdl) )`，
`IDF(t) = ln( (N − n_t + 0.5) / (n_t + 0.5) + 1 )`。
分词用 `jieba.lcut`，查询词去重后求和。返回按分降序，`topk` 截断。

### `retrieval/bge_encoder.py` — B 负责 `[STUB]`

```python
class BgeEncoder:
    def encode_docs(self, docs: list[dict]) -> "np.ndarray":   # (N, d)，已 L2 归一化
    def encode_query(self, query: str) -> "np.ndarray":        # (d,)
class FaissIndex:
    def build(self, doc_ids: list[str], embs) -> None: ...
    def search(self, q_emb, topk: int) -> list[tuple[str, float]]: ...
```

**注意**：向量必须 **L2 归一化**，用内积索引（`IndexFlatIP`）等价余弦相似度。
BGE 系列中文模型建议 query 前加指令前缀（`bge-small-zh-v1.5` 可省略，写成配置项）。

### `retrieval/rrf.py` — B 负责 `[STUB]`

```python
def fuse(runs: list[list[tuple[str, float]]],
         weights: list[float] | None = None,
         k: int = 60) -> list[tuple[str, float]]: ...
```

**公式**：`score(d) = Σ_i w_i / (k + rank_i(d))`，`rank` 从 **1** 起。
`d` 只在某一路出现时，该路缺失项贡献 0（**不要**按缺失补 rank）。
`weights` 缺省为等权；E2 的 α = 稠密路权重，BM25 路权重取 `1 − α`。

### `rerank/cross_encoder.py` — C 负责 `[STUB]`

```python
class Reranker:
    def rerank(self, query: str, candidates: list[tuple[str, float]],
               topn: int) -> list[tuple[str, float]]: ...
```

`candidates` 是上一级检索的 `(doc_id, score)` 列表；返回**新的** `(doc_id, score)`
（score 为相关性 logit 或概率，仅用于排序，不与其他路可比）。

### `reader/context_builder.py` — C 负责 `[STUB]`

```python
def build_context(doc_ids: list[str], doc_map: dict[str, str],
                  order: str = "natural",
                  max_chars: int = 2000,
                  gold_doc_id: str | None = None
                  ) -> tuple[str, list[str]]: ...
```

`order` 取值：`natural`（保持检索顺序）、`gold_first`、`gold_middle`、`gold_last`。
**E5 只改这个函数的 `order`**，其他一律不动。

**返回值是二元组** `(context_text, used_doc_ids)`：`used_doc_ids` 是真正进入上下文的段落。
E5 必须靠它区分两种失败——「金标段落根本没被拼进去（截断/排序问题）」与
「金标进去了但阅读器没答对（模型问题）」，这是位置偏差实验的结论前提。

截断策略固定为「按段落整段拼入，超 `max_chars` 即停」——不许截半个段落，
否则 F1 的差异会混入「句子被切断」这个混淆因素。

### `reader/extractive_qa.py` — C 负责 `[STUB]`

```python
class ExtractiveReader:
    def answer(self, question: str, context: str) -> tuple[str, float]: ...
```

返回 `(answer_text, score)`；无答案返回 `("", 0.0)`。

### `eval/metrics.py` — D 校验扩充 `[READY]`（A 已给参考实现）

```python
def recall_at_k(ranked: list[str], gold: str, k: int) -> float
def mrr(ranked: list[str], gold: str) -> float
def ndcg_at_k(ranked: list[str], gold: str, k: int) -> float
def em(pred: str, ref: str) -> float
def f1(pred: str, ref: str) -> float
def bootstrap_ci(values: list[float], n: int = 1000, alpha: float = 0.05,
                 seed: int = 42) -> tuple[float, float]
```

`ranked` 是**按相关度降序**的 `doc_id` 列表（长度可小于 k）。
`em` / `f1` 内部做归一化：小写、去中英文标点、中文去空格。
**全组只用这一份指标代码**，不许各自手写，否则报告里会出现两套口径。

---

## 3. 实验结果格式

所有实验统一写 `results/raw/<run_id>.json`：

```json
{"run_id": "E2_rrf_alpha0.5",
 "config": {"retriever": "rrf", "alpha": 0.5, "topk": 5, "rerank": false},
 "meta": {"split": "smoke", "n_questions": 500, "seed": 42,
          "git_commit": "abc1234", "python": "3.10.11"},
 "metrics": {"recall@1": 0.71, "recall@5": 0.91, "mrr": 0.79,
             "ndcg@5": 0.83, "f1": 0.68,
             "latency_ms": 41.2, "recall@5_ci": [0.89, 0.93]},
 "per_question": [{"qid": "dev_0001", "ranked": ["p_00312", "p_00401"],
                   "gold": "p_00312", "hit": true, "pred": "……", "f1": 0.80}]}
```

- `run_id` 命名：`E{编号}_{配置摘要}`，例：`E4_rerank50`、`E3_chunk256`
- 图文件名与实验编号对齐：`results/fig/E2_alpha_recall.png`
- 表格：`results/tab/E1_baseline.csv`

---

## 4. 协作规矩

| 规矩 | 说明 |
|---|---|
| 配置 | 所有超参走 `config.yaml`，**禁止硬编码**（报告第 4 章要贴这张表） |
| 随机种子 | 统一 `seed=42`，bootstrap 也要固定 |
| 依赖 | 新增依赖必须写进 `requirements.txt` 并说明版本 |
| 日志 | 真实跑实验写 `results/log/<run_id>.log`（耗时、显存、环境版本） |
| 指标 | 只调 `eval/metrics.py` |
| 大文件 | 数据集与 `results/raw/full_*.json` 不进 git |

---

## 5. 谁改了什么，记在这里

| 日期 | 谁 | 改了什么 | 影响 |
|---|---|---|---|
| 11-17 | A | 冻结初版接口 | 全员 |