# 踩坑日志

> 所有人随时追加，一行一条。报告第 4 章「实现过程」直接从这里取材。
> 格式：`[日期] [谁] 现象 → 原因 → 解法`

## 环境

- [11-17] [A] `python eval/run_e1.py --dry-run` 在纯净环境报 `ModuleNotFoundError: eval` → 直接用 `python eval/run_e1.py` 从仓库根目录跑就会把仓库根加进 `sys.path`；不要 `cd eval` 后跑。
- [11-17] [A] Windows 下 `python` 是 Microsoft Store 的别名，指向空壳 → 用 `py -3` 或装官方 python 后 `where python` 确认。
- [11-21] [D] 实测本机 `python` / `python3` / `py` **三个别名全是 0 字节占位符**（`C:\Users\LEGION\AppData\Local\Microsoft\WindowsApps\`），运行直接报 `The system cannot find the path specified`；`py -3` 同样失效 → 真实解释器在 `C:\Users\LEGION\AppData\Local\Python\bin\python.exe`（Python 3.14.5），全组统一用它跑。注意 3.14 装 torch/faiss 可能没有 wheel，B/C 建议装 3.10/3.11 单独建 venv，数据脚本不受影响（纯标准库）。
- [11-17] [B] `pip install faiss-cpu` 一开始失败（`PermissionError: [Errno 13]`，`pip-unpack` 临时目录 DACL 被继承成 0o700；`files.pythonhosted.org` 直链挂起、tuna 的 `/packages/...` 直连 403）→ **后来装上了**：`pip install --no-cache-dir --only-binary :all: -i https://pypi.tuna.tsinghua.edu.cn/simple --target <libs> faiss-cpu` 拿到 1.16.0，`import faiss` 正常。实测 3507×512 规模下 faiss 单查询 0.20–0.21 ms、numpy 0.27–0.28 ms，两后端 top-100 **集合 100% 一致、逐位 99.75% 相同**（float32 内积并列时的次序交换），端到端指标逐位一致 → 所以 `FaissIndex` 保留双后端，换机器装不上也不影响结论。
- [11-17] [B] `pip install torch` 250 MB 轮子在镜像下反复超时（hf/tuna 都不稳），且同样撞上面的沙箱权限错 → **整条链路改走 onnxruntime + 官方 ONNX 权重**（`models/bge-small-zh-v1.5-onnx/model_int8.onnx`，22.8 MB）。代价：得自带 BERT WordPiece 分词器（`retrieval/bert_tokenizer.py`），且是 int8 量化精度。CPU 上单查询 2 ms。（torch 与 CUDA 的版本对应因此在本项目里用不上，这一项不必再补。）
- [11-18] [B] fp32 `model.onnx`（94.9 MB）单连接只有 40–130 KB/s 且频繁断 → **改用 4 连接分段续传**（`scripts/_dl.py`，HEAD 取 Content-Length + `Range` 分片）后 320 KB/s 完成。量化对照结论：单路稠密 fp32 0.9531 / int8 0.9487（−0.44pt），**融合后 R@5 两者都是 0.9854、MRR 差 0.08pt**，正式口径用 int8 不影响主结论。
- [11-17] [B] `pip install jieba` 只有 sdist，`--only-binary :all:` 装不了，且 tuna 的 `../../packages/...` 推断 URL 直连 403 → 解法：把 PyPI JSON 里 sdist 的 host 从 `files.pythonhosted.org` 换成 `pypi.tuna.tsinghua.edu.cn`，19 MB 秒下，`tarfile` 解包到 `.pylibs\jieba`。
- [11-18] [B] `pytest` 的 `tmp_path` 建在用户 Temp 下，被沙箱拒绝（`PermissionError`）→ 测试里落盘一律用 `ROOT/.cache/tests/`，不要用 `tmp_path`。
- [ ] 待记：8G 显存下 reranker 的 batch 上限实测（C 补）

## 数据

- [11-21] [D] CMRC2018 官方主下载是 CodaLab 的 zip（要注册），速度不稳 → 改用 `ymcui/cmrc2018` 仓库里的 **squad-style json**（GitHub raw 直下，train 7.4 MB / dev 3.3 MB / trial 0.8 MB），无需解压，字段和 SQuAD 一致。
- [11-21] [D] `answer_start` 不能直接信：squad-style dev 有 **3 题**偏移（`context[start:start+len]` 对不上） → 一律用 `context.find(answer_text)` 重算，偏移数与"答案不在段落"数分别记账（`data/raw/build_stats.json`），`eval/check_data.py` 会把两者都断言为 0。
- [11-21] [D] CMRC dev 每题带 **3 个答案**（通常内容相同）→ 取 `answers[0]`；不要取平均长度或最长答案，否则和 SQuAD 口径不一致。
- [11-21] [D] 一题多段落的情况在 CMRC dev **不存在**（848 段 / 3219 题，每题唯一金标段落，同段被多题复用）→ 单金标假设成立，`nDCG` 的 IDCG 恒为 1。
- [11-21] [D] 段落去重：train/dev/trial 之间**零重叠**（3507 = 2403 + 848 + 256）→ 语料库直接合并三份，不必担心金标段落同时出现在问题和语料的不同 id 上。
- [11-21] [D] `data/*.jsonl`、`data/raw/` 都在 `.gitignore` 里 → 换机器要重跑 `build_data.py`；**脚本进 git、数据不进**，别人 clone 后一条命令重建。
- [11-21] [B] 本机 `raw.githubusercontent.com` 被 hosts 拦死，`curl -L` 三个 squad-style json 全失败 → 改走 GitHub Blob API：`api.github.com/repos/ymcui/cmrc2018/git/trees/master?recursive=1` 取 `squad-style-data/*.json` 的 blob sha，再 `git/blobs/<sha>` 取 base64 解码落盘（`data/raw/cmrc2018_{train,dev,trial}.json`，与 D 的校验数完全一致：3507 段 / 3219 题 / 3 题位置校正）。**换机器若 GitHub API 也被墙，可让 D 直接带 json 过来。**
- [11-21] [B] **doc_id 编号方案一变，向量缓存会静默错配**：第 1 棒 D 的数据从 `p_00000` 起改成 `p_00001` 起，条目数仍是 3507，旧的 `emb_docs_*_n3507.npz` 按「n 相同」被命中，于是旧向量被贴到新 doc_id 上，指标会偏但没有报错 → 解法是缓存键加**语料内容 sha1 摘要**（`FaissIndex.build_from_encoder`，文件名带 `-<12 位摘要>`，npz 里也存 `digest`，命中时再核一次）。改数据之后先删 `.cache/emb/*` 再重跑。

## 检索

- [11-17] [B] **分词器换了，结论会反过来**：同一套 E2 配置，用"汉字单字 + 二元组"时 α 曲线单调下降（看着像"融合没用"），换 jieba 精确分词后变成"中间高两头低"（α=0.2 最优）→ 分词器必须写进 `config.yaml` 并随模型表一起冻结；BM25 的 `to_dict/load` 里存了分词器配置，否则加载后分数会静默错位。
- [11-18] [B] 稠密向量忘了 L2 归一化时（裸内积）：单路 Recall@5 从 0.9487 掉到 **0.9366（−1.2pt）**、MRR 0.9207→0.8972（−2.4pt）；α=0.2 融合后 Recall@5 0.9854→**0.9857（+0.03pt，噪声内）**、MRR 0.9628→0.9604（−0.24pt）→ 归一化必须做；有意思的是**融合把召回损失兜住了，但精度损失会转移到排序指标（MRR）上**，只看 Recall 会误判"没影响"。
- [11-18] [B] RRF 的 k 取 60 与其他值的差别：k∈{10,20,60,100,200} 对应 R@5 = 0.9832 / 0.9848 / **0.9854** / 0.9848 / 0.9823 → 差 0.3 个点以内，k=60 不是必须，但 60 附近最稳；k 越小越"赢者通吃"，k 越大越平权。
- [11-18] [B] 稠密候选深度 `depth`：20/50/100/200 → R@5 = 0.9845 / 0.9851 / 0.9854 / 0.9857 → ≥50 基本饱和，取 100 是保守值（多算 100 个内积约 +0.1 ms）。
- [11-18] [B] ONNX 编码器内部默认会 L2 归一化，外部 `normalize=False` 会被它吞掉 → 做归一化 ablation 时给 `ONNXBGEEncoder` 加 `normalize` 参数透传，否则两臂结果完全相同（踩过一次，白跑 1 小时）。向量缓存的 stamp 也必须带上 `norm/raw`，否则会命中另一臂的缓存。
- [11-21] [B] 目录里同时放 `model.onnx`（fp32）与 `model_int8.onnx` 时，`--encoder auto` 会**优先 fp32**，E2/E3 的数字会比报告的 int8 表高约 0.4 个点（单路稠密）→ 复现必须在命令里钉死 `--model-path models/bge-small-zh-v1.5-onnx/model_int8.onnx`；`BgeEncoder` 的 `model_path` 既接受目录也接受 `.onnx` 文件路径（`_pick_backend` 有分支认后缀）。
- [11-21] [B] 测延迟必须先**预热一轮**：faiss 首次调用要建索引对象/热身，不预热时同一份代码会得出"numpy 比 faiss 快"的相反结论（0.266 vs 0.278 ms）；预热 + 5 轮取中位数后是 faiss 0.21 ms / numpy 0.28 ms。凡是写进报告的延迟数字都要注明「预热 + N 轮中位数 + 机器负载」。

## 重排与上下文

- [11-22] [C] **重排延迟与吞吐必须分开测**：`bge-reranker-base`（278M）在 CPU 上单线程约 0.3–0.4 s/pair（512 token），depth=100 时单查询要几十秒；同一份代码用 batch=16 + 多线程算指标，换成 batch=1 + 单线程报延迟，两者差 3–5 倍。混在一起写进报告会被当成"重排不可用"。→ 代码里 `Reranker(threads=1, score_threads=18)`，`benchmark_latency()` 临时切回单线程。
- [11-22] [C] **int8 ONNX 的怪现象**：`Xenova/bge-reranker-base` 的 `onnx/model_int8.onnx` 在 **1 线程**下 0.74 s/pair，比 fp32 torch（0.28 s/pair）还慢；把 `intra_op_num_threads` 提到 18 后降到 0.14 s/pair（2 倍快于 fp32）→ onnxruntime 的量化 kernel 依赖多线程，单线程口径下不能拿 int8 当"更快"的论据。正式数字仍用 fp32，int8 只做一致性对照（附录 C.3）。
- [11-22] [C] **阅读器不滑窗 = 一半答案在窗外**：CMRC 段落均值 509 字 ≈ 700 token，`max_length=512` 直接截断会把段落后半段（常常就是答案位置）切掉。用 `truncation="only_second"` + `doc_stride=128` 切多窗，再取全局最高分 span。
- [11-22] [C] **抽取式 QA 的假阳性**：不过滤 `sequence_ids == 1` 时模型可能返回问题里的词（问题与答案共享实体），F1 会虚高；必须只接受 context 侧的 token。
- [11-22] [C] **位置实验与截断的混淆**：`max_chars=2000` 装不下 5 段（均值 509 字），`gold_last` 常被"超限即停"挡在门外，F1 掉到 0 看着像"尾部位置效应巨大"，其实是"答案没进上下文"。→ `build_context` 返回 `used_doc_ids` 并旁路记录 `gold_used`，E5 同时报 `gold_kept_rate` 与 `gold_kept_f1`，另跑 `max_chars=4000`（全装得下）做纯位置对照。
- [11-22] [C] **模型权重的源，逐个试过**（本机实测）：`huggingface.co` 直连 502；`hf-mirror.com` 必须带 `User-Agent`，而且**文件下载会 302 到 `cas-bridge.xethub.hf.co`**——那个节点在本机 0–25 KB/s 甚至完全不通（这才是 407 MB 下不来的真正原因，不是"整体带宽差"）；`hf-mirror.net`（另一个域名）**直出文件、不跳 xethub**，1–4 MB/s，4 连接分段 40 秒下完 160 MB；`modelscope.cn` 单连接 7–12 MB/s 但只有部分仓库有镜像（Xenova 的 BGE 系列有，`uer/roberta-base-chinese-extractive-qa` 没有）。→ 大文件先 `curl -I` 看 `Location`：跳 xethub 就换 `hf-mirror.net`；有 modelscope 镜像就走 modelscope（`scripts/_fetch.py`）。
- [11-22] [C] `ai.gitcode.com/hf_mirrors/...` 虽然页面能打开，但 `resolve/`、`-/raw/` 全部返回 HTML（要 JS/登录），不能当直链用；`mirror.sjtu.edu.cn`、`hf-api.gitee.com`、`aifasthub.com`、`hf.llm.vin`、`moeyy.xyz` 在本机全部 404/握手失败。别再浪费时间试这些。
- [11-22] [C] **`_dl.py` 的分段续传在 TimeoutError 后会重复写**：重试沿用了进入函数时算好的 `Range` 起点，若上一次已写入 N 字节再失败，重试会从同一位置再写一遍，part 文件体积超过理论份额 → 拼接出来的权重加载失败。跨「不同 `--parts` 次数」复用 part 文件也会错位。→ 换 parts 数或中断后**先删 part 文件**再下；下完必须 `onnxruntime.InferenceSession` / `torch.load` 验一次大小。
- [ ] 待记：8G 显存下 reranker 的 batch 上限实测（本机无 GPU，C 未测）

## 呈现

- [11-18] [B]（原「待记：matplotlib 中文字体（D 补）」的实测答案，D 可直接用）matplotlib 在 Windows 上中文变方框：先把 `MPLCONFIGDIR` 指到仓库外（默认缓存在 `C:\Users\<you>\AppData\Local\matplotlib`，沙箱下不可写），再让 `font.sans-serif` 首选 `Microsoft YaHei`（其次 SimHei / SimSun），`axes.unicode_minus=False`；写不出中文就自动退英文标签，不要留方框进报告。
- [ ] 待记：PPT 里插入的图分辨率要求（E 补）

## 环境认知偏差

- [11-24] [A] **"没有 GPU" 是一次集体误判**。复核演示环境时发现，A 的机器其实就是计划里那块 **RTX 4070 Laptop（8 GB，驱动 610.47）**；但从清华源 `pip install torch` 拿到的是 **CPU 版**（`2.15.0+cpu`，189 MB，`torch.version.cuda=None`），因此 `torch.cuda.is_available()=False`，`config.yaml` 的 `device: cuda` 会**静默退回 cpu**（只打印一行 `[reader] 没有可用的 cuda，自动退回 cpu`）。叠加另两条：B 因 250 MB 轮子反复超时压根没装 torch（见 #11）、C 的机器确实无可用 GPU——三者合起来让全组都以为"我们没有 GPU"。
  **教训**：`没用上` 与 `没有` 是两件事。报告里必须写"未启用 GPU / 延迟为 CPU 口径"；写成"没有 GPU"等于对硬件条件做了虚假陈述，评审一问就露。要真用上：装 CUDA 版 torch（`--index-url https://download.pytorch.org/whl/cu121`，约 2.5 GB），但**延迟会与 E4 的 CPU 口径不可比**，所以只在演示时用，不进结论。
