"""把 B 撰写的两章 markdown 转成可直接并入小组报告的 Word 文档。

排版按要求：正文宋体小四（12pt）、行距 1.5、首行缩进 2 字符、标题加粗黑体、
表格用三线表风格（Word 里是普通表格 + 细边框），图按 6.5 英寸宽插入。
只处理本组这两章支持的 markdown 子集：#/##/### 标题、段落、- 列表、| 表格 |、
![alt](path) 图片、``` 代码块、> 引用。
"""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

REPO = Path(r"D:\dsh\写报告的地方\rag-qa-ablation")
OUT = REPO / "report" / "分工B_报告章节.docx"

SOURCES = [
    ("第2章_相关工作.md", None),
    ("附录B_检索实验记录.md", None),
]


def set_cn_font(run, name="宋体", size=12, bold=False):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), name)
    rfonts.set(qn("w:hAnsi"), name)
    rfonts.set(qn("w:eastAsia"), name)


def add_para(doc, text, size=12, bold=False, font="宋体", indent=True,
             align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=0, space_after=6,
             line_spacing=1.5):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = line_spacing
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.alignment = align
    if indent:
        pf.first_line_indent = Pt(size * 2)
    run = p.add_run(text)
    set_cn_font(run, font, size, bold)
    return p


def add_heading(doc, text, level):
    sizes = {1: 16, 2: 14, 3: 12.5, 4: 12}
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(12 if level <= 2 else 8)
    pf.space_after = Pt(6)
    pf.line_spacing = 1.3
    run = p.add_run(text)
    set_cn_font(run, "黑体", sizes.get(level, 12), bold=True)
    return p


def add_table(doc, rows):
    cols = max(len(r) for r in rows)
    table = doc.add_table(rows=len(rows), cols=cols)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j in range(cols):
            cell = table.cell(i, j)
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_after = Pt(2)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(row[j] if j < len(row) else "")
            set_cn_font(run, "宋体", 10.5, bold=(i == 0))
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return table


def add_image(doc, path: Path):
    if not path.exists():
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(path), width=Cm(16.5))


def inline_runs(p, text, size=12):
    """处理 **粗体**、`代码` 与 ^[n]^ 上标引用。"""
    for token in re.split(r"(\*\*[^*]+\*\*|`[^`]+`|\^\[[^\]]+\]\^)", text):
        if not token:
            continue
        if token.startswith("**") and token.endswith("**"):
            run = p.add_run(token[2:-2])
            set_cn_font(run, "宋体", size, bold=True)
        elif token.startswith("`") and token.endswith("`"):
            run = p.add_run(token[1:-1])
            set_cn_font(run, "Consolas", size - 1)
        elif token.startswith("^[") and token.endswith("]^"):
            run = p.add_run(token[1:-1])
            set_cn_font(run, "宋体", size)
            run.font.superscript = True
        else:
            run = p.add_run(token)
            set_cn_font(run, "宋体", size)


def add_rich_para(doc, text, size=12, indent=True, space_after=6, line_spacing=1.5):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = line_spacing
    pf.space_before = Pt(0)
    pf.space_after = Pt(space_after)
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if indent:
        pf.first_line_indent = Pt(size * 2)
    inline_runs(p, text, size)
    return p


def convert(md_path: Path, doc: Document):
    lines = md_path.read_text(encoding="utf-8").splitlines()
    i = 0
    in_code = False
    code_buf: list[str] = []
    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            if in_code:
                for cl in code_buf:
                    p = doc.add_paragraph()
                    p.paragraph_format.line_spacing = 1.0
                    p.paragraph_format.space_after = Pt(0)
                    p.paragraph_format.left_indent = Pt(12)
                    run = p.add_run(cl if cl else " ")
                    set_cn_font(run, "Consolas", 9.5)
                code_buf = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code_buf.append(raw)
            i += 1
            continue

        if not stripped:
            i += 1
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m:
            add_heading(doc, m.group(2).strip(), len(m.group(1)))
            i += 1
            continue

        m = re.match(r"^!\[[^\]]*\]\(([^)]+)\)\s*$", stripped)
        if m:
            target = m.group(1)
            if target.startswith("http"):
                i += 1
                continue
            add_image(doc, (md_path.parent / target).resolve())
            i += 1
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in row if c):
                    rows.append(row)
                i += 1
            if rows:
                add_table(doc, rows)
            continue

        if re.match(r"^[-*]\s+", stripped):
            text = re.sub(r"^[-*]\s+", "", stripped)
            p = add_rich_para(doc, text, indent=False, space_after=3, line_spacing=1.4)
            p.paragraph_format.left_indent = Pt(24)
            p.paragraph_format.first_line_indent = Pt(-12)
            i += 1
            continue

        if re.match(r"^\d+\.\s+", stripped):
            p = add_rich_para(doc, stripped, indent=False, space_after=3, line_spacing=1.4)
            p.paragraph_format.left_indent = Pt(24)
            p.paragraph_format.first_line_indent = Pt(-18)
            i += 1
            continue

        if stripped.startswith(">"):
            text = stripped.lstrip("> ").strip()
            p = add_rich_para(doc, text, size=10.5, indent=False, space_after=4,
                              line_spacing=1.3)
            p.paragraph_format.left_indent = Pt(18)
            i += 1
            continue

        if set(stripped) <= set("-—*_ ") and len(stripped) >= 3:
            i += 1
            continue

        add_rich_para(doc, stripped)
        i += 1


def main():
    doc = Document()
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(2.8)
    section.right_margin = Cm(2.8)
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)

    style = doc.styles["Normal"]
    style.font.name = "宋体"
    style.font.size = Pt(12)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")

    for name, _ in SOURCES:
        md = REPO / "report" / name
        if md.exists():
            convert(md, doc)
            doc.add_page_break()

    doc.save(OUT)
    print(f"[docx] {OUT}")
    print(f"[docx] paragraphs={len(doc.paragraphs)} tables={len(doc.tables)}")


if __name__ == "__main__":
    main()
