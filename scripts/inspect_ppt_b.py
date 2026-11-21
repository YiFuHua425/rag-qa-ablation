from pptx import Presentation

p = Presentation(r"D:\dsh\写报告的地方\rag-qa-ablation\ppt\检索部分_B.pptx")
W, H = p.slide_width, p.slide_height
print("slide", W / 914400, "x", H / 914400, "slides", len(p.slides))
for i, s in enumerate(p.slides, 1):
    over = []
    for sh in s.shapes:
        if sh.left is None:
            continue
        if (sh.left < 0 or sh.top < 0 or sh.left + sh.width > W + 1000
                or sh.top + sh.height > H + 1000):
            over.append((sh.name, round(sh.left / 914400, 2), round(sh.top / 914400, 2),
                         round((sh.left + sh.width) / 914400, 2),
                         round((sh.top + sh.height) / 914400, 2)))
    texts = [sh.text_frame.text[:40].replace("\n", " / ") for sh in s.shapes
             if sh.has_text_frame and sh.text_frame.text.strip()]
    print(f"--- slide {i}: shapes={len(s.shapes)} overflow={over}")
    for t in texts[:7]:
        print("   ", t)
