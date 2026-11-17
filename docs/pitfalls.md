# 踩坑日志

> 所有人随时追加，一行一条。报告第 4 章「实现过程」直接从这里取材。
> 格式：`[日期] [谁] 现象 → 原因 → 解法`

## 环境

- [11-17] [A] `python eval/run_e1.py --dry-run` 在纯净环境报 `ModuleNotFoundError: eval` → 直接用 `python eval/run_e1.py` 从仓库根目录跑就会把仓库根加进 `sys.path`；不要 `cd eval` 后跑。
- [11-17] [A] Windows 下 `python` 是 Microsoft Store 的别名，指向空壳 → 用 `py -3` 或装官方 python 后 `where python` 确认。
- [ ] 待记：faiss-cpu 在 Windows 下的 pip 直装是否成功（B 补）
- [ ] 待记：torch 版本与 CUDA 版本对应（B/C 补）
- [ ] 待记：8G 显存下 reranker 的 batch 上限实测（C 补）

## 数据

- [ ] 待记：CMRC2018 下载地址与解压后的目录结构（D 补）
- [ ] 待记：dev 集里是否存在一题多段落的情况，怎么处理的（D 补）
- [ ] 待记：`answer_text` 与 `answer_start` 对不上的样本有多少（跑 `check_data.py` 看）

## 检索

- [ ] 待记：向量忘了归一化时 Recall 掉多少（B 补，这是个好报告点）
- [ ] 待记：RRF 的 k 取 60 与其他值的差别（B 补）

## 重排与上下文

- [ ] 待记：重排 depth 增大到多少时延迟开始不可接受（C 补）
- [ ] 待记：`max_chars` 截断把金标段落切掉的题目比例（C 补，E5 要用）

## 呈现

- [ ] 待记：matplotlib 中文字体（Windows 用 `SimSun` / `Microsoft YaHei`）（D 补）
- [ ] 待记：PPT 里插入的图分辨率要求（E 补）