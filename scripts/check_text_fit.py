"""粗略检查 PPT 文本是否溢出所在文本框（无渲染器时的替代检查）。

估算方法：按字号折算每行可容纳的中文字符数（中文字宽 ≈ 1em，西文 ≈ 0.55em），
行高 ≈ 1.35 * 字号，行数 = ceil(总宽 / 框宽)，与框高比较，给出"估算行数/可容纳行数"。
"""
from pptx import Presentation
from pptx.util import Emu

EMU_PER_IN = 914400
PATH = r"D:\dsh\写报告的地方\rag-qa-ablation\ppt\检索部分_B.pptx"

prs = Presentation(PATH)
worst = []
for idx, slide in enumerate(prs.slides, 1):
    for sh in slide.shapes:
        if not sh.has_text_frame or not sh.text_frame.text.strip():
            continue
        tf = sh.text_frame
        box_w_in = (sh.width or Emu(0)) / EMU_PER_IN
        box_h_in = (sh.height or Emu(0)) / EMU_PER_IN
        total_lines = 0.0
        max_size = 0
        for para in tf.paragraphs:
            text = "".join(r.text for r in para.runs)
            if not text:
                total_lines += 0.4
                continue
            size = max((r.font.size.pt if r.font.size else 18) for r in para.runs)
            max_size = max(max_size, size)
            width_em = sum(1.0 if ord(c) > 0x2E80 else 0.55 for c in text)
            cpl = max(1.0, (box_w_in - 0.16) * 72 / size)
            total_lines += max(1, -(-width_em // cpl))
        need_h = total_lines * max_size * 1.35 / 72
        ratio = need_h / box_h_in if box_h_in else 0
        if ratio > 1.05:
            worst.append((idx, round(ratio, 2), round(need_h, 2), round(box_h_in, 2),
                          tf.text[:34].replace("\n", "/")))
worst.sort(key=lambda r: -r[1])
print("疑似文本溢出（ratio>1.05）：" if worst else "文本容量估算：全部文本框未超容")
for row in worst:
    print(f"  slide {row[0]}  ratio={row[1]}  需要 {row[2]}in / 框高 {row[3]}in  «{row[4]}»")
