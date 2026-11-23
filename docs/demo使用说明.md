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

## 1. 现在的环境状态（已就绪，两台机器都能跑完整链路）

2026-11-23 在 A 的机器上装完了全部依赖与权重，实测跑通含重排与 F1 的完整链路：

| 项 | 状态 |
|---|---|
| 解释器 | `C:\Users\20102\AppData\Local\Programs\Python\Python312\python.exe`（3.12.10，终端里的 `python`） |
| 依赖 | torch 2.15.0 / transformers 5.24.0 / onnxruntime 1.31.1 / jieba 0.42.1 / pyyaml 6.0.3 |
| 权重 | `models/bge-small-zh-v1.5-onnx`（23 MB）、`models/bge-reranker-base`（1082 MB）、`models/roberta-base-chinese-extractive-qa`（406,799,727 字节） |
| 向量缓存 | `.cache/emb/emb_demo_paragraph_onnx-...n3507.npz`（命中后 47 ms） |

实测耗时：**首次编码全库 55 s（只一次）→ 之后完整链路 15 s**（重排 8.3 s + 阅读器 1.0 s + 检索 0.1 s）。

```powershell
cd D:\git\机器学习.汇报

python demo.py --alpha 0.2 --rerank              # 完整链路，约 15 秒
python demo.py --alpha 0.2                       # 跳过重排，约 6 秒
python demo.py --index 3 --rerank                # 指定第 4 题
python demo.py --list                            # 看有哪些题
python demo.py --question "苹果公司的创始人是谁"    # 现场自定义提问（无金标）
python demo.py --score-threads 8 --rerank        # 限线程（默认 cpu_count-2） 
```

自动选题是**两遍扫描**：先只在「答案 ≤18 字」的题里找（长答案 F1 天生低），找不到才放宽；
优先级是「一路没召回→融合救回」>「融合比两路都好」>「两路分歧大」。

预期结尾（实测结果，题目 `dev_2414`）：

```
【8】小结（这张表就是全篇的论证结构）
  配置                金标名次
  ------------------------
  BM25 单路              1
  稠密单路             未召回
  RRF 融合               5
  重排后                 1

  → 实际用于回答的是「重排后」，金标名次 1。
     重排把金标从融合后的第 5 位顶到第 1 位 —— 精排在这一题上决定性地救回来了。
```

这一题把全篇论证演全了：BM25 命中 → 稠密漏召 → 融合被稀释到第 5 → 精排救回第 1，
最后阅读器抽出 `高桥俊昌`，**F1 = 1.0000，EM = 1**。

---

## 2. 换机器 / C 的机器：从零装一遍

C 的机器还留着 E4/E5 时期的权重，可直接跳到 2.2；新机器照下面走（含这次踩到的两个坑）。
这段也是**验收 C 那三个模块**的最短路径：`rerank/cross_encoder.py`、`reader/context_builder.py`、
`reader/extractive_qa.py` 会被真实调用。

### 2.1 准备（一次性）

```powershell
cd <你的仓库路径>
git pull                                    # 拿到 demo.py

# ① 依赖（约 200 MB，实测 4–8 分钟；torch 走清华源可能冷缓存较慢）
pip install torch transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install onnxruntime pyyaml              # 若还没有
pip install jieba                           # ⚠ 不要加 --only-binary :all:，它只有 sdist

# ② 数据：jsonl 不入 git，必须本机重建
python data/build_data.py --sample 20
python eval/check_data.py --strict          # 四项断言必须全过
python eval/check_data.py --make-smoke 500

# ③ 三个权重（合计约 1.5 GB，实测 2–3 分钟）
python scripts/download_model.py --parts 1                                  # 稠密编码器 23 MB
python scripts/download_models_c.py --source modelscope --only rerank       # 1082 MB，13.6 MB/s
python scripts/download_models_c.py --source hf --only reader --parts 4     # 406.8 MB，9.2 MB/s
```

**踩坑 1：reader 分段下载会留错位的 part 文件。**
若中断过或重试过，`models/roberta-base-chinese-extractive-qa/` 里会剩下
`pytorch_model.bin.part0..3`，四段加起来 **大于** 406.8 MB（我这次是 468 MB）。
合并会得到坏文件。务必先删再下：

```powershell
del models\roberta-base-chinese-extractive-qa\*.part*
python scripts/download_models_c.py --source hf --only reader --parts 4
dir models\roberta-base-chinese-extractive-qa\pytorch_model.bin   # 必须是 406,799,727 字节
```

**踩坑 2：`pip install ... --only-binary :all:` 会让整批安装失败。**
jieba 只有 sdist，pip 解析阶段就会整体报错，连 torch 都不会装。所以 jieba 单独装。

装完自检：

```powershell
python -c "import torch,transformers,onnxruntime,jieba;print(torch.__version__,transformers.__version__)"
```

### 2.2 跑完整版

```powershell
python demo.py --rerank --alpha 0.2
```

多出来的两段长这样：

```
【5】重排：交叉编码器（bge-reranker-base）　　只重排前 20 个候选
  名次  doc_id          重排分   金标
  1     p_03052        -2.0147   ★
  2     p_01733        -4.2118
  → 金标名次：1
  耗时 8316 ms（22 线程算分，416 ms/候选）
  注：报告里的延迟口径是 CPU 单线程 ≈225 ms/候选（见 E4），此处为吞吐口径

【7】抽取式阅读器（在上下文里定位答案片段）
  预测答案 '高桥俊昌'　score 19.6349　1011 ms
  标准答案 '高桥俊昌'
  F1 1.0000　EM 1
```

> 重排默认用 `cpu_count-2` 线程算分（与 `run_e4` 同一公式）。只用 1 线程时这一步会从
> 8.3 s 涨到 26 s——之前 demo 就是这个毛病，已修。用 `--score-threads N` 可覆盖。

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