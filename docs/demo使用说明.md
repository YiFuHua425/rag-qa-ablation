# demo.py 使用说明

> 面向两个人看：**A（集成/录屏）** 与 **C（重排与阅读器作者，本机有 torch 与权重）**。
> 脚本位置：仓库根 `demo.py`。一条命令把「检索 → 融合 → 重排 → 拼装 → 抽取」摊开。

## 0. 这个 demo 能证明什么

它不是"又一个评测脚本"，而是**把消融结论可视化到终端**：同一个问题，金标在
`BM25 / 稠密 / RRF 融合 / 重排` 四套配置下各排第几，最后阅读器抽出哪几个字。

| 输出段 | 对应的报告内容 |
|---|---|
| 【1】分词 | 第 3 章 3.2 BM25 的词项与 idf |
| 【2】稀疏路 | E2 的 BM25 单路 |
| 【3】稠密路 | E1 基线 + 第 3 章 3.3 双塔 |
| 【4】RRF 融合 | 第 3 章 3.4 + E2 的 α 扫描 |
| 【5】重排（可选） | 第 3 章 3.5 + E4 的 Pareto |
| 【6】上下文拼装 | 第 3 章 3.7 + E5 的截断/位置混淆 |
| 【7】阅读器 | 第 3 章 3.6 + E4/E5 的 F1 |
| 【8】小结 | 第 5 章的总表逻辑 |

---

## 1. A（颜福洪）的机器：只演示检索侧，约 5 秒

你的机器没有 torch，**阅读器与重排会自动跳过**，前面七段照常出数——这对录屏已经够用
（论证结构全在检索侧）。

```powershell
cd D:\git\机器学习.汇报

# 一次性准备（已做过可跳过）
python scripts/download_model.py --parts 1        # 稠密编码器 int8，22.8 MB

# 每次演示
python demo.py --alpha 0.2                        # 自动挑一题，约 5 秒
python demo.py --index 3                          # 指定第 4 题
python demo.py --list                             # 看有哪些题
python demo.py --question "苹果公司的创始人是谁"    # 现场自定义提问（无金标）
```

> 首次运行会编码 3507 段（约 55 秒）并写入 `.cache/emb/`；第二次起命中缓存，5 秒内跑完。
> **录屏前先跑一遍预热。**

预期结尾：

```
【8】小结（这张表就是全篇的论证结构）
  配置                金标名次
  ------------------------
  BM25 单路              3
  稠密单路               5
  RRF 融合               2
  → 实际用于回答的是「RRF 融合」，金标名次 2。
     融合把最好的单路从第 3 位提升到第 2 位 —— 这正是混合检索的收益。
```

---

## 2. C（王文超）的机器：完整链路，含重排与 F1

你的机器有 torch + `bge-reranker-base` + 阅读器权重（跑 E4/E5 时已下过），
所以**只有你能跑出第【5】段重排和第【7】段的 F1/EM 真数字**——这正是你最该验证的部分：
`rerank/cross_encoder.py`、`reader/context_builder.py`、`reader/extractive_qa.py` 三个模块被真实调用。

### 2.1 准备（一次性）

```powershell
cd <你的仓库路径>
git pull                                   # 拿到 demo.py（若还没有）

# ① 数据：jsonl 不入 git，必须本机重建
python data/build_data.py --sample 20
python eval/check_data.py --strict          # 四项断言必须全过
python eval/check_data.py --make-smoke 500

# ② 稠密编码器（22.8 MB）—— 你之前复用 B 的候选缓存，可能没有这个权重
python scripts/download_model.py --parts 1

# ③ 你已有的权重：确认在
dir models
#   bge-reranker-base\                （约 1.1 GB）
#   bge-small-zh-v1.5-onnx\           （22.8 MB，上一步下的）
#   roberta-base-chinese-extractive-qa\ （约 407 MB）
```

若哪个缺，补齐：

```powershell
python scripts/download_models_c.py --source modelscope --only rerank
python scripts/download_models_c.py --source hf --only reader --parts 16
```

### 2.2 跑完整版

```powershell
python demo.py --rerank --alpha 0.2
```

多出来的两段长这样：

```
【5】重排：交叉编码器（bge-reranker-base）　　只重排前 20 个候选
  名次  doc_id          重排分   金标
  1     p_01809         0.8412
  2     p_02433         0.7735   ★
  → 金标名次：2
  耗时 3620 ms（CPU 单线程 ≈ 181 ms/候选）

【7】抽取式阅读器（在上下文里定位答案片段）
  预测答案 '拜占庭式建筑风格'　score 12.4471　444 ms
  标准答案 '拜占庭式建筑风格'
  F1 1.0000　EM 1
```

### 2.3 你该重点看什么

| 检查点 | 期望 | 若不符说明 |
|---|---|---|
| 【5】重排分单调递减 | 是 | 打分/排序有 bug |
| 【5】金标名次 ≤ 融合前 | 通常更好 | `depth` 太小或 doc_map 没设 |
| 【6】`used_doc_ids` 与 `build_context` 返回一致 | 是 | `get_last_stats()` 没清空 |
| 【7】预测答案是**上下文里的连续子串** | 必须 | 滑窗 offset 还原错 |
| 【7】答案不出现在问题里 | 必须 | `sequence_ids==1` 过滤失效 |

最后一条是你在 `docs/pitfalls.md` 里记过的坑，这个 demo 现在能一眼看出来。

---

## 3. 输出怎么读（两人通用）

```
【0】装配     语料规模 / 编码器 / 索引耗时（首次 55s，之后 77ms）
【1】分词     查询被切成哪些词（中文分词会改变结论，见 pitfalls）
【2】稀疏路   BM25 前 k + 金标名次
【3】稠密路   BGE 前 k + 金标名次
【4】融合     两路名次 → 融合分 → 融合后名次（只看排名，不看分数）
【5】重排     仅 --rerank 时出现
【6】拼装     几段进上下文、被预算挡掉哪些、金标在第几段
【7】阅读器   预测答案 / 标准答案 / F1 / EM
【8】小结     四个配置下的金标名次对照 ← 这一张表就是全篇论点
```

---

## 4. 常见报错

| 现象 | 原因 | 解法 |
|---|---|---|
| `ModuleNotFoundError: eval` | 在 `eval/` 里跑，或不在仓库根 | `cd` 到仓库根再 `python demo.py` |
| `[跳过] 阅读器不可用：需要 torch + transformers` | 没装 torch | 正常现象（A 的机器就这样）；要完整版就装 |
| `[跳过] 重排不可用` | 同上 | 少了 `--rerank` 也能演示检索侧 |
| 首次跑 55 秒 | 编码全库 | 正常；第二次命中 `.cache/emb/` 后 5 秒 |
| 每次都要 55 秒 | 缓存没命中（tag 或语料摘要变了） | 删 `.cache/emb/` 重跑一次，之后稳定命中 |
| 中文乱码 | 终端编码 | 脚本已强制 UTF-8 输出；用 Windows Terminal 打开 |
| `找不到 qid=...` | 题号不属于当前 split | `--list` 查，或换 `--split full` |

---

## 5. 录屏建议（给 E 参考）

1. 先跑 `python demo.py --alpha 0.2` 预热缓存，正式录时 5 秒出结果；
2. 一段一提词：**【2】看 BM25 排第 3 → 【3】看稠密排第 5 → 【4】融合到第 2**，这就是"混合检索有用"的证据；
3. 若用 C 的机器录【5】【7】，记得补一句"重排把 R@1 拉高 9 个点，但延迟从 0.45 s 涨到 4 s"，与 E4 的 Pareto 图对上；
4. 结尾停在【8】那张表上，正好接第 5 章总表。