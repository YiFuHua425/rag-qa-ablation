# 交接单 · D（数据与评测）

> 依据 `执行顺序-接力棒与交接标准.md` 逐棒交付。下一位接手的人（A 集成 / E 统稿）只看本页。
> 数据侧细节见 `data/README.md`；评测口径见 `report/第4章_实现过程.md`。

## 0. D 在第几棒

| 棒次 | 该棒要求 D 做什么 | 完成判据（原文） | 状态 |
|---|---|---|---|
| **第 1 棒** | CMRC2018 清洗成 `corpus.jsonl` / `dev.jsonl`；`check_data.py --strict` 四项断言全绿 | 越界数 = 0、gold 缺失 = 0、抽 20 条人工核对 | ✅ 3507 段 / 3219 题 / 848 金标段 / 越界 0 / 校正 3 题 |
| **第 4 棒** | 所有 run 统一指标 + bootstrap CI + 图表口径 | 每个 run 有 json + CI + 能对应一张图 | ✅ 指标口径由 D 与 A 共同定义；CI 全部 1000 次 seed=42 |
| **第 7 棒** | 第 4 章「实现过程」素材（环境/超参/复现/踩坑） | 报告 3 页 + 附录 | ✅ 素材已交，`report/第4章_实现过程.md` 由 A 汇编定稿 |

## 1. 【第 1 棒交接】交棒人：D

```
交付物：data/build_data.py（跨 train/dev/trial 去重 + answer_start 校正 + 统计留档）
        data/README.md（全套数字与重建步骤）
        data/corpus.jsonl（3507 段）、data/dev.jsonl（3219 题）、data/dev_smoke.jsonl（500 题）
验收方式：python data/build_data.py --sample 20
        python eval/check_data.py --strict            → 四项断言全过（exit 0）
        python eval/check_data.py --make-smoke 500
已知限制：jsonl 与 raw 均进 .gitignore，换机器必须重跑（一条命令，约 1 分钟）；
        CMRC dev 每题 3 个答案，只取 answers[0]（与 SQuAD 口径一致），若三者有差异 F1 会略偏保守。
下一棒是谁：B（建索引）、C（阅读器）、A（pipeline）
```

**关键数字**（可复算，留档于 `data/raw/build_stats.json`）：

| 项 | 值 |
|---|---|
| 语料段落 | 3507（train 2403 + dev 848 + trial 256，三份零重叠） |
| 评测问题 | 3219 |
| 金标段落 | 848（每题唯一，同段被多题复用） |
| 段落长度 | 均值 509.1 字 / 最长 980 字 |
| 答案长度 | 均值 10.7 字 |
| `answer_start` 校正 | 3 题 |
| 答案越界 / 金标缺失 | 0 / 0 |

## 2. 【第 4 棒交接】交棒人：D（与 A 共同定口径）

```
交付物：results/tab/*.csv 的列定义与统计口径；主结果总表的口径说明（report/主结果总表.md 第 0 节）
验收方式：任一 run 的 metrics 都应含 reca11@k_ci（bootstrap 1000 次）；重跑同配置差值 < 0.1 点
已知限制：split 不一致——E1 smoke(500)、E2/E3/E6 full(3219)、E4/E5 smoke(100)；
        跨实验比较必须带 split 与 n，E1 的正式基线用 E2_dense 行（配置等价）。
下一棒是谁：E（第 5 章）
```

**评测口径（写进报告「评测协议」小节）**：

| 项 | 定义 |
|---|---|
| Recall@k | 前 k 名是否含金标；单金标 → {0,1} |
| MRR | $1/\mathrm{rank}(d^\*)$ |
| nDCG@k | $1/\log_2(\mathrm{rank}+1)$，单相关故 IDCG=1 |
| EM / F1 | NFKC + 去标点 + 去空白后，字符级计算 |
| 置信区间 | `bootstrap_ci(n=1000, seed=42, alpha=0.05)` |
| 延迟 | CPU 单线程、batch=1、预热后 3 轮中位数 |

## 3. 数据侧的六个决定（第 4 章已展开）

1. **不信 `answer_start`，用 `context.find()` 重算** —— squad-style dev 有 3 题偏移；
2. **段落跨 split 去重后统一编号**（train → dev → trial，先到先得）；
3. **一题多答案取 `answers[0]`**，与 SQuAD 口径一致；
4. **单金标假设成立**（848 段 / 3219 题，无"一题多段落"），故 nDCG 的 IDCG 恒为 1；
5. **数据不进 git、脚本进 git** —— 换机器一条命令重建同一份数据；
6. **`check_data.py` 把约束写成断言** —— 防止"用错数据跑完所有实验"这类不可逆事故。

## 4. 已知遗留

| # | 事项 | 影响 | 建议 |
|---|---|---|---|
| 1 | 图表样式未由 D 统一（B 用 `eval/plots.py`、C 用 `eval/plots_c.py` 两套脚本） | 报告里图风格略不一致 | 统稿前由 D 跑一次 `plots_c.py` 覆盖 E2/E3 两张图，或接受现状（三张图配色已在同一色系） |
| 2 | E4/E5 只在 smoke 100 题上跑过 | CI 约 ±2–4 点 | 时间允许优先补 E5 full（只依赖检索缓存 + 阅读器） |
| 3 | `build_stats.json` 在 `data/raw/`（gitignored） | 换机器需重跑才有 | 报告引用的数字已固化在 `data/README.md`，不影响复现 |

## 5. 一键验收

```powershell
python data/build_data.py --sample 20     # 重建数据
python eval/check_data.py --strict        # 四项断言（必须 exit 0）
python -m pytest tests -q                 # 24 项（B 13 + C 11）
```