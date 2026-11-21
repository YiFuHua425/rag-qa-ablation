# data/ —— 数据目录（负责人：D）

**本目录的 jsonl 不进 git**（见 `.gitignore`）。每个人本地自己放，格式必须一致。

## 状态：已就绪（第 1 棒交付）

```
python data/build_data.py --sample 20     # 从 data/raw 生成
python eval/check_data.py --strict        # 四项断言全绿
python eval/check_data.py --make-smoke 500
```

| 项 | 数值 |
|---|---|
| `corpus.jsonl` | **3507** 段（train 2403 + dev 848 + trial 256，跨集无重复） |
| `dev.jsonl` | **3219** 题，全部保留（0 题被丢弃） |
| `dev_smoke.jsonl` | **500** 题，`seed=42` 等距随机抽样 |
| 金标段落数 | 848（dev 每题唯一金标，同一段落多题共用） |
| 段落长度 | 均值 509.1 字，最长 980 字（≤ 500 字左右的阅读器上限之外，E3 切块实验有意义） |
| 答案长度 | 均值 10.7 字 |
| `answer` 越界 / `answer_start` 不符 | **0 / 0** |

**原始标注修正**：squad-style 的 `answer_start` 有 **3 题**偏移，`build_data.py` 用
`context.find(answer_text)` 重算并记账（`data/raw/build_stats.json`）。报告第 4 章「数据清洗」直接引用这个数。

**一题多答案**：CMRC2018 dev 每题给 3 个答案，内容通常相同，取 `answers[0]`。

**人工核对**：`data/raw/manual_check_sample.txt` 已导出 20 条（等距抽样，带金标段落上下文），
逐条核对结论：**20/20 答案均为回答该问题的片段**。

## 要产出什么

| 文件 | 内容 | 谁用 |
|---|---|---|
| `corpus.jsonl` | 全部段落，`{"doc_id","text"}` | B（建索引）、A（pipeline） |
| `dev.jsonl` | 全部测试问题，`{"qid","question","gold_doc_id","answer_text","answer_start"}` | 全员 |
| `dev_smoke.jsonl` | 固定种子抽 500 题，`check_data.py --make-smoke 500` 生成 | 开发迭代用 |

格式硬约束见 `INTERFACE.md` 第 1 节。**`answer_text` 必须是其金标段落的子串**，
这条不过 `check_data.py --strict` 就一直红。

## 首选：CMRC2018

中文阅读理解数据集，每题自带**答案跨度**，正好用来做"金标段落即检索标签"。
下载后把 train + dev + trial 的段落合并去重成 `corpus.jsonl`，dev 的问题做成 `dev.jsonl`。

**下载**（squad-style 版，直接是 json，不用解压；约 11 MB，比 CodaLab 包省事）:

```bash
mkdir -p data/raw
for s in train dev trial; do
  curl -L -o data/raw/cmrc2018_$s.json \
    https://raw.githubusercontent.com/ymcui/cmrc2018/master/squad-style-data/cmrc2018_$s.json
done
```

**生成**:

```bash
python data/build_data.py --sample 20     # → data/corpus.jsonl / data/dev.jsonl
# 体检（数据没到位时会提示缺什么）
python eval/check_data.py
# 到位后必须全绿，否则不要开始跑实验
python eval/check_data.py --strict
# 切 500 题的 smoke 子集
python eval/check_data.py --make-smoke 500
```

## 备选：SQuAD v2（英文）

如果 CMRC 下载不顺利 **8 小时内换掉，不要硬扛**。SQuAD v2 格式最规整，
字段映射关系：

| SQuAD 字段 | 我们的字段 |
|---|---|
| `data[].paragraphs[].context` | `text`（doc_id 自己编号） |
| `data[].paragraphs[].qas[].question` | `question` |
| `qas[].answers[0].text` | `answer_text` |
| `qas[].answers[0].answer_start` | `answer_start` |
| 该 paragraph 的编号 | `gold_doc_id` |

指标脚本与 pipeline **完全通用**，不用改一行代码——只是报告里数据来源换成英文，
标题里的"中文问答"要相应改成"问答"。

## 目录约定

```
data/
  corpus.jsonl
  dev.jsonl
  dev_smoke.jsonl
  raw/            # 原始下载解压，随便乱放（已 ignore）
```