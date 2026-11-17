# data/ —— 数据目录（负责人：D）

**本目录的 jsonl 不进 git**（见 `.gitignore`）。每个人本地自己放，格式必须一致。

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
下载后把 train + dev 的段落合并去重成 `corpus.jsonl`，dev 的问题做成 `dev.jsonl`。

```bash
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