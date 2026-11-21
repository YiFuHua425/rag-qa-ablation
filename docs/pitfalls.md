# 踩坑日志

> 所有人随时追加，一行一条。报告第 4 章「实现过程」直接从这里取材。
> 格式：`[日期] [谁] 现象 → 原因 → 解法`

## 环境

- [11-17] [A] `python eval/run_e1.py --dry-run` 在纯净环境报 `ModuleNotFoundError: eval` → 直接用 `python eval/run_e1.py` 从仓库根目录跑就会把仓库根加进 `sys.path`；不要 `cd eval` 后跑。
- [11-17] [A] Windows 下 `python` 是 Microsoft Store 的别名，指向空壳 → 用 `py -3` 或装官方 python 后 `where python` 确认。
- [11-21] [D] 实测本机 `python` / `python3` / `py` **三个别名全是 0 字节占位符**（`C:\Users\LEGION\AppData\Local\Microsoft\WindowsApps\`），运行直接报 `The system cannot find the path specified`；`py -3` 同样失效 → 真实解释器在 `C:\Users\LEGION\AppData\Local\Python\bin\python.exe`（Python 3.14.5），全组统一用它跑。注意 3.14 装 torch/faiss 可能没有 wheel，B/C 建议装 3.10/3.11 单独建 venv，数据脚本不受影响（纯标准库）。
- [ ] 待记：faiss-cpu 在 Windows 下的 pip 直装是否成功（B 补）
- [ ] 待记：torch 版本与 CUDA 版本对应（B/C 补）
- [ ] 待记：8G 显存下 reranker 的 batch 上限实测（C 补）

## 数据

- [11-21] [D] CMRC2018 官方主下载是 CodaLab 的 zip（要注册），速度不稳 → 改用 `ymcui/cmrc2018` 仓库里的 **squad-style json**（GitHub raw 直下，train 7.4 MB / dev 3.3 MB / trial 0.8 MB），无需解压，字段和 SQuAD 一致。
- [11-21] [D] `answer_start` 不能直接信：squad-style dev 有 **3 题**偏移（`context[start:start+len]` 对不上） → 一律用 `context.find(answer_text)` 重算，偏移数与"答案不在段落"数分别记账（`data/raw/build_stats.json`），`eval/check_data.py` 会把两者都断言为 0。
- [11-21] [D] CMRC dev 每题带 **3 个答案**（通常内容相同）→ 取 `answers[0]`；不要取平均长度或最长答案，否则和 SQuAD 口径不一致。
- [11-21] [D] 一题多段落的情况在 CMRC dev **不存在**（848 段 / 3219 题，每题唯一金标段落，同段被多题复用）→ 单金标假设成立，`nDCG` 的 IDCG 恒为 1。
- [11-21] [D] 段落去重：train/dev/trial 之间**零重叠**（3507 = 2403 + 848 + 256）→ 语料库直接合并三份，不必担心金标段落同时出现在问题和语料的不同 id 上。
- [11-21] [D] `data/*.jsonl`、`data/raw/` 都在 `.gitignore` 里 → 换机器要重跑 `build_data.py`；**脚本进 git、数据不进**，别人 clone 后一条命令重建。

## 检索

- [ ] 待记：向量忘了归一化时 Recall 掉多少（B 补，这是个好报告点）
- [ ] 待记：RRF 的 k 取 60 与其他值的差别（B 补）

## 重排与上下文

- [ ] 待记：重排 depth 增大到多少时延迟开始不可接受（C 补）
- [ ] 待记：`max_chars` 截断把金标段落切掉的题目比例（C 补，E5 要用）

## 呈现

- [ ] 待记：matplotlib 中文字体（Windows 用 `SimSun` / `Microsoft YaHei`）（D 补）
- [ ] 待记：PPT 里插入的图分辨率要求（E 补）