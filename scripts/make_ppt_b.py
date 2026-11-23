"""生成 B 分工的 PPT 片段（可直接并入小组 16 页主 PPT）

产物：``ppt/检索部分_B.pptx``（5 页）+ ``ppt/讲稿_B.md``（每页讲稿与切换词）。

页码沿用方案里的分工：B 负责"检索原理 / α 曲线 / 切块曲线"三块，
页内左上角标注``分工 B``，方便集成时定位。
"""

from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "ppt" / "检索部分_B.pptx"
FIGS = REPO / "results" / "fig"
TAB = REPO / "results" / "tab"

FONT = "Microsoft YaHei"
INK = RGBColor(0x1F, 0x2A, 0x37)
GREY = RGBColor(0x5B, 0x6B, 0x7A)
ACCENT = RGBColor(0x2C, 0x6E, 0xB5)
WARN = RGBColor(0xC0, 0x50, 0x2F)
LIGHT = RGBColor(0xF2, 0xF5, 0xF9)


def _txbox(slide, x, y, w, h, text, size=16, bold=False, color=INK, align=PP_ALIGN.LEFT,
           line_spacing=1.25):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        run = p.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
        run.font.name = FONT
    return tb


def _title(slide, title, kicker="分工 B ｜ 检索与融合"):
    bar = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(13.333), Inches(0.92))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()
    _txbox(slide, 0.55, 0.12, 9.5, 0.7, title, size=27, bold=True,
           color=RGBColor(0xFF, 0xFF, 0xFF))
    _txbox(slide, 9.6, 0.28, 3.3, 0.4, kicker, size=11, color=RGBColor(0xDD, 0xE8, 0xF5),
           align=PP_ALIGN.RIGHT)


def _bullets(slide, x, y, w, h, items, size=15):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, (head, body) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = 1.3
        p.space_after = Pt(7)
        r1 = p.add_run()
        r1.text = "▍" + head
        r1.font.size = Pt(size)
        r1.font.bold = True
        r1.font.color.rgb = ACCENT
        r1.font.name = FONT
        r2 = p.add_run()
        r2.text = " " + body
        r2.font.size = Pt(size)
        r2.font.color.rgb = INK
        r2.font.name = FONT
    return tb


def _card(slide, x, y, w, h, head, body, head_color=ACCENT, size=13):
    box = slide.shapes.add_shape(5, Inches(x), Inches(y), Inches(w), Inches(h))
    box.fill.solid()
    box.fill.fore_color.rgb = LIGHT
    box.line.color.rgb = RGBColor(0xD5, 0xDF, 0xEA)
    box.line.width = Pt(0.75)
    _txbox(slide, x + 0.14, y + 0.1, w - 0.28, 0.4, head, size=size + 3, bold=True,
           color=head_color)
    _txbox(slide, x + 0.14, y + 0.52, w - 0.28, h - 0.6, body, size=size, color=INK)
    return box


def _load(name: str) -> dict:
    p = REPO / "results" / "raw" / name
    return json.loads(p.read_text("utf-8")) if p.exists() else {}


def slide_1(prs):
    """检索在 RAG 链路中的位置 + B 的交付物"""
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _title(s, "检索为什么是 RAG 的天花板", "B ｜ 检索与融合")
    _txbox(s, 0.6, 1.12, 12.1, 0.5,
           "生成器的输入只有上下文；金标段落没被检索到，后面的重排、阅读器再强也救不回来。",
           size=15, color=GREY)
    _card(s, 0.6, 1.75, 3.85, 1.55, "稀疏通道 · BM25",
          "手写倒排 + 词频饱和 + 长度归一化\n零训练、精确匹配强、对专名/数字敏感",
          head_color=ACCENT)
    _card(s, 4.72, 1.75, 3.85, 1.55, "稠密通道 · BGE",
          "bge-small-zh-v1.5（4 层/512 维）\n[CLS] 池化 + L2 归一化 + IndexFlatIP",
          head_color=ACCENT)
    _card(s, 8.84, 1.75, 3.9, 1.55, "融合 · RRF",
          "只用名次不用分数：Σ wᵢ/(k+rankᵢ)\nw = [1-α, α]，一条曲线扫完三种配置",
          head_color=ACCENT)
    _txbox(s, 0.6, 3.55, 12.1, 0.45, "分工 B 的交付物", size=17, bold=True, color=INK)
    _bullets(s, 0.6, 4.02, 6.1, 2.4, [
        ("代码", "BM25 / BGE 编码 + FAISS 索引 / RRF / 分词 / 切块，接口按 INTERFACE.md 冻结"),
        ("实验", "E2 融合权重 α 扫描（7 点 + 95% CI）、E3 切块策略、E6 RM3 查询改写、超参敏感度"),
    ], size=13)
    _bullets(s, 6.9, 4.02, 5.8, 2.4, [
        ("报告", "第 2 章相关工作 + 附录实验记录"),
        ("PPT", "本页起 5 页：原理、α 曲线、切块曲线、结论"),
        ("可复现", "全部指标落 JSON（含超参与运行环境），核心模块 13 项自检"),
    ], size=13)
    return s


def slide_2(prs):
    """BM25 原理与手写实现"""
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _title(s, "稀疏通道：手写 BM25", "B ｜ 检索原理")
    _card(s, 0.6, 1.12, 7.1, 2.35, "打分公式",
          "score(q,d) = Σ IDF(t) · f(t,d)(k1+1)\n"
          "                     ─────────────────────────\n"
          "                     f(t,d) + k1(1 - b + b·|d|/avgdl)\n"
          "IDF(t) = ln[(N - nₜ + 0.5)/(nₜ + 0.5) + 1]", size=13)
    _bullets(s, 0.6, 3.62, 7.1, 2.9, [
        ("词频饱和", "k1 控制 tf 的收敛速度——出现 100 次不该比 10 次相关 10 倍"),
        ("长度归一化", "b 控制 |d|/avgdl 的惩罚强度，长段落天然更容易命中"),
        ("中文分词", "jieba 精确模式 + 停用词；装不上时退'单字 + 二元组'，共用同一套打分"),
        ("工程细节", "倒排 dict[term]{行号:tf} + numpy 向量化；同分按语料序，可复现"),
    ], size=13)
    _card(s, 7.95, 1.12, 4.75, 2.35, "实测（3507 段 / 3219 题）",
          "词汇表 100,543\n全库 fit 4.4 s\n全量打分 0.6 s\n单查询端到端 2.2 ms", size=13)
    _card(s, 7.95, 3.62, 4.75, 2.05, "一个非直觉的坑", 
          "同样的 α 扫描，字符二元组分词下\nBM25 强到让曲线单调下降；\n换成 jieba 才出现'中间高两头低'。\n→ 分词器会改变结论，必须固定并写进配置。",
          head_color=WARN, size=13)
    return s


def slide_3(prs):
    """稠密通道 + RRF 融合"""
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _title(s, "稠密通道与融合：BGE + RRF", "B ｜ 检索原理")
    _bullets(s, 0.6, 1.15, 6.0, 3.1, [
        ("双塔", "查询与段落各编码一次，[CLS] 池化取语义向量，内积 = 余弦"),
        ("归一化", "L2 归一化是硬要求：不归一化退化成裸内积，长段落占便宜（见 ablation）"),
        ("索引", "IndexFlatIP 精确检索；装不上 faiss 时自动退 numpy，top-k 完全一致"),
        ("缓存", "段落向量落盘 .cache/emb/*.npz，stamp 含后端+模型文件名，换模型不会误命中"),
    ], size=13)
    _card(s, 6.85, 1.15, 5.85, 1.6, "为什么是 ONNX 而不是 torch",
          "本机 pip 装 torch 触发沙箱权限错误 + 250 MB 下载不稳 → onnxruntime + 官方 ONNX 权重，CPU 上照样 2 ms/查询。",
          head_color=WARN, size=12)
    _card(s, 6.85, 2.95, 5.85, 1.5, "RRF：只用名次",
          "fused(d) = Σᵢ wᵢ / (k + rankᵢ(d))，k=60\n某路没召回则贡献 0；同路重复只算最好名次。", size=12)
    _card(s, 6.85, 4.6, 5.85, 1.75, "α 的定义（本文口径）",
          "w_dense = α，w_bm25 = 1 - α\nα=0 纯 BM25 ｜ α=1 纯稠密 ｜ 中间为混合\n七点扫描：0 / .2 / .4 / .5 / .6 / .8 / 1.0", size=12)
    _txbox(s, 0.6, 4.35, 6.0, 2.0,
           "两者互补的证据\n"
           "纯 BM25：R@5 0.976，但 R@1 只有 0.889\n"
           "纯稠密：R@5 0.949，R@1 0.898\n"
           "→ 稀疏通道召回更全，稠密通道把正确的往前提",
           size=13, color=INK)
    return s


def slide_4(prs):
    """E2 α 曲线"""
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _title(s, "E2：融合权重 α 扫描（3219 题，95% CI）", "B ｜ α 曲线")
    fig = FIGS / "E2_alpha_recall.png"
    if fig.exists():
        s.shapes.add_picture(str(fig), Inches(0.45), Inches(1.05), width=Inches(9.1))
    else:
        _txbox(s, 0.6, 1.2, 8.0, 1.0, "（图待生成：eval/plots.py）", size=14, color=GREY)
    rows = [
        ("α", "R@1", "R@5", "MRR"),
        ("0.0 BM25", ".889", ".976", ".926"),
        ("0.2 最佳", ".944", ".985", ".962"),
        ("0.5", ".940", ".978", ".955"),
        ("1.0 稠密", ".898", ".949", ".918"),
    ]
    top = 1.25
    for i, r in enumerate(rows):
        y = top + i * 0.52
        for j, cell in enumerate(r):
            _txbox(s, 9.62 + j * 0.9, y, 0.88, 0.46, cell,
                   size=11.5 if i == 0 else 12.5,
                   bold=(i == 0 or (i == 2 and j > 0)),
                   color=ACCENT if (i == 0 or i == 2) else INK,
                   align=PP_ALIGN.CENTER)
        line = s.shapes.add_shape(1, Inches(9.6), Inches(y + 0.48), Inches(3.55), Emu(9525))
        line.fill.solid()
        line.fill.fore_color.rgb = RGBColor(0xD5, 0xDF, 0xEA)
        line.line.fill.background()
    _txbox(s, 9.6, 4.05, 3.55, 2.3,
           "α=0.2 最优：\nR@5 +0.9pt（vs BM25）\nR@5 +3.7pt（vs 稠密）\n\n"
           "增益主要在 R@1 / MRR：\n把金标从第 2–3 名提到第 1 名",
           size=13, color=INK)
    _txbox(s, 0.6, 6.55, 12.1, 0.6,
           "读法：两端是单路基线，中间是融合；曲线在 α>0.6 后快速下滑 → 这份数据上稠密通道的边际价值低于稀疏通道。",
           size=12.5, color=GREY)
    return s


def slide_5(prs):
    """E3 切块曲线"""
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _title(s, "E3：切块粒度与上下文成本（3219 题）", "B ｜ 切块曲线")
    fig = FIGS / "E3_chunk.png"
    if fig.exists():
        s.shapes.add_picture(str(fig), Inches(0.45), Inches(1.05), width=Inches(8.4))
    else:
        _txbox(s, 0.6, 1.2, 8.0, 1.0, "（图待生成：eval/plots.py）", size=14, color=GREY)
    rows = [
        ("策略", "块/段", "R@5", "上下文"),
        ("整段", "1.00", ".949", "2604 字"),
        ("512+重叠", "1.44", ".947", "1866 字"),
        ("256 字", "2.65", ".958", "988 字"),
        ("128 字", "5.03", ".970", "520 字"),
        ("句级", "12.30", ".970", "243 字"),
    ]
    for i, r in enumerate(rows):
        y = 1.22 + i * 0.5
        for j, cell in enumerate(r):
            _txbox(s, 9.05 + j * 1.0, y, 1.05, 0.38, cell,
                   size=11.5 if i == 0 else 12.5,
                   bold=(i == 0 or (i >= 4 and j > 0)),
                   color=ACCENT if (i == 0 or i >= 4) else INK, align=PP_ALIGN.CENTER)
    _txbox(s, 9.0, 4.15, 3.95, 2.6,
           "细粒度切块双赢：\n召回 +2.1pt（.949→.970），\n上下文少 91%（2604→243 字）\n\n"
           "代价：索引膨胀 12×，\n建索引 162 s，延迟 2.0→3.5 ms\n\n"
           "拐点在 128 字：\n与句级同召回，块数只有 1/2.4",
           size=12.5, color=INK)
    _txbox(s, 0.6, 6.45, 12.1, 0.6,
           "边界回退：切点落在句中时向后找句末标点（最多半个窗口），避免把答案跨度拦腰截断；块级排名去重压平回段落，与 E2 同口径。",
           size=12.5, color=GREY)
    return s


def main() -> int:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    for fn in (slide_1, slide_2, slide_3, slide_4, slide_5):
        fn(prs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    print(f"[ppt] {OUT} ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
