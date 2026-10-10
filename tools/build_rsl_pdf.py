"""Build an RSL-compliant PDF from the RSL markdown.

- Times New Roman, 3 cm margins, single spacing
- Equations (1)(2)(3) rendered as mathtext images with right-aligned numbers
- Robust single-pass table parse (no duplicates)
- Figure 1 embedded after Table 4
"""
import re, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xml.sax.saxutils import escape
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image, KeepTogether)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors

FONT_DIR = r"C:/Windows/Fonts"
pdfmetrics.registerFont(TTFont("Times", os.path.join(FONT_DIR, "times.ttf")))
pdfmetrics.registerFont(TTFont("Times-Bold", os.path.join(FONT_DIR, "timesbd.ttf")))
pdfmetrics.registerFont(TTFont("Times-Italic", os.path.join(FONT_DIR, "timesi.ttf")))
pdfmetrics.registerFont(TTFont("Times-BoldItalic", os.path.join(FONT_DIR, "timesbi.ttf")))
pdfmetrics.registerFontFamily("Times", normal="Times", bold="Times-Bold",
                              italic="Times-Italic", boldItalic="Times-BoldItalic")

MD = r"D:/ace/source/MDYOLO_RSL_letter.md"
FIG = r"D:/ace/source/RSL_fig1_domain_shift.png"
PDF = r"D:/ace/source/MDYOLO_RSL_letter_submitted_v2.pdf"
EQ_DIR = r"D:/ace/source/tools/_eq"

CONTENT_W = A4[0] - 2 * cm * 3  # content width in points

# ----- equation rendering (mathtext, no LaTeX install needed) -----
EQUATIONS = {
    1: r"$p_x = g_x + (2\sigma(t_x)-1)\cdot s,\; p_y = g_y + (2\sigma(t_y)-1)\cdot s,\; p_w = s\,e^{t_w},\; p_h = s\,e^{t_h}$",
    2: r"$m = \sigma(c)^{\alpha}\cdot \mathrm{IoU}^{\beta}$",
    3: r"$L_{\mathrm{VFL}} = -\alpha_v\, q\,[\,q\log p + (1\!-\!q)\log(1\!-\!p)\,] \;-\; \alpha_v\, p^{\gamma}\log(1\!-\!p)$",
    4: r"$L_{\mathrm{WIoU}} = 1 - \exp\!\left(-\dfrac{\Delta x^{\gamma} + \Delta y^{\gamma}}{W_g^{\gamma} + H_g^{\gamma}}\right)\cdot \mathrm{IoU}$",
}
EQ_IMG = {}
os.makedirs(EQ_DIR, exist_ok=True)
for n, tex in EQUATIONS.items():
    path = os.path.join(EQ_DIR, f"eq{n}.png")
    fig = plt.figure(figsize=(7.5, 0.9))
    fig.text(0.5, 0.5, tex, ha="center", va="center", fontsize=17)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    EQ_IMG[n] = path

# ----- styles -----
def style(name, **kw):
    base = dict(fontName="Times", fontSize=10.5, leading=14, alignment=TA_JUSTIFY)
    base.update(kw)
    return ParagraphStyle(name, **base)

S_TITLE = style("t", fontName="Times-Bold", fontSize=16, leading=20,
                alignment=TA_CENTER, spaceAfter=4)
S_AUTHOR = style("a", fontSize=11, leading=14, alignment=TA_CENTER, spaceAfter=2)
S_H1 = style("h1", fontName="Times-Bold", fontSize=13, leading=16,
             spaceBefore=12, spaceAfter=6)
S_H2 = style("h2", fontName="Times-Bold", fontSize=11.5, leading=14,
             spaceBefore=8, spaceAfter=4)
S_BODY = style("b", spaceAfter=6)
S_ABSTRACT = style("abs", spaceAfter=6)
S_CAP = style("cap", fontName="Times-Italic", fontSize=9, leading=11,
              alignment=TA_CENTER, spaceBefore=2, spaceAfter=8)
S_CODE = ParagraphStyle("code", fontName="Courier", fontSize=8.5, leading=11,
                        leftIndent=10, backColor=colors.HexColor("#f2f2f2"),
                        borderPadding=4, spaceAfter=6)
S_REF = style("ref", fontSize=9.5, leading=12, leftIndent=18,
              firstLineIndent=-18, spaceAfter=3)
S_EQNUM = style("eqn", fontSize=10.5, leading=14, alignment=TA_CENTER,
                fontName="Times")

def md_inline(text):
    t = escape(text)
    t = re.sub(r'`([^`]+)`', r'<font face="Courier">\1</font>', t)
    t = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', t)
    t = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<i>\1</i>', t)
    # superscripts: x^α  x^β  x^γ  (only letter/greek/digit after caret)
    t = re.sub(r'\^\s*([A-Za-z\u03b1-\u03c9\u0391-\u03a9\u00b2\u00b3\u00b9])',
               r'<super>\1</super>', t)
    return t

# ----- read -----
with open(MD, encoding="utf-8") as f:
    lines = f.read().replace("\u2212", "-").split("\n")

story = []
abstract_started = False
in_ref = False
in_code = False

def add_table(header, body, cap=None):
    data = [[Paragraph(md_inline(c), style("th", fontName="Times-Bold",
                                          fontSize=8.5, leading=10)) for c in header]]
    for row in body:
        data.append([Paragraph(md_inline(c), style("tc", fontSize=8.5, leading=10))
                     for c in row])
    ncol = len(header)
    widths = [CONTENT_W / ncol] * ncol
    tbl = Table(data, colWidths=widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#888888")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    block = [tbl, Spacer(1, 4)]
    if cap:
        block.append(Paragraph(md_inline(cap), S_CAP))
    # keep each table with its caption from splitting across pages
    story.append(KeepTogether(block))

def add_equation(n):
    from PIL import Image as PILImage
    px_w, px_h = PILImage.open(EQ_IMG[n]).size
    # equations are rendered at 200 dpi; reportlab defaults to 1px=1pt,
    # so set the natural physical size explicitly (px/200 inch -> *72 pt)
    img = Image(EQ_IMG[n], width=px_w * 72.0 / 200.0, height=px_h * 72.0 / 200.0)
    img._restrictSize(12 * cm, 3 * cm)
    img.hAlign = "CENTER"
    num = Paragraph("(%d)" % n, S_EQNUM)
    tbl = Table([[img, num]], colWidths=[12 * cm, 2 * cm])
    tbl.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                             ("ALIGN", (1, 0), (1, 0), "RIGHT")]))
    story.append(tbl)
    story.append(Spacer(1, 4))

def add_figure():
    try:
        from PIL import Image as PILImage
        w, h = PILImage.open(FIG).size
        iw = 5.8 * inch
        ih = iw * h / w
        img = Image(FIG, width=iw, height=ih)
        img.hAlign = "CENTER"
        story.append(img)
        story.append(Paragraph(
            "Figure 1. Cross-domain localization degradation from VisDrone "
            "(in-domain) to UAVDT-M (zero-shot). (a) Mean best per-image IoU "
            "drops from 0.40 to 0.17. (b) Image-level overlap rates at "
            "IoU \u2265 0.25 and IoU \u2265 0.50 collapse across the domain shift.",
            S_CAP))
    except Exception as e:
        print("figure embed skipped:", e)

EQ_RE = re.compile(r"\(\s*(\d+)\s*\)\s*$")

i = 0
while i < len(lines):
    s = lines[i].strip()
    if s == "" or s in ("---", "***"):
        i += 1; continue
    # fenced code block
    if s.startswith("```"):
        i += 1; buf = []
        while i < len(lines) and not lines[i].strip().startswith("```"):
            buf.append(lines[i]); i += 1
        i += 1
        story.append(Paragraph(escape("\n".join(buf)).replace("\n", "<br/>"),
                               S_CODE))
        continue
    # table block (header + separator + body)
    if (s.startswith("|") and i + 1 < len(lines)
            and lines[i + 1].strip().startswith("|")
            and set(lines[i + 1].strip().replace(" ", "")) <= set("|-:")):
        tbl = []
        j = i
        while j < len(lines) and lines[j].strip().startswith("|"):
            tbl.append(lines[j].strip()); j += 1
        rows = [[c.strip() for c in r.strip("|").split("|")] for r in tbl]
        i = j
        # consume optional blank lines then the caption
        cap = None
        while i < len(lines) and lines[i].strip() == "":
            i += 1
        if i < len(lines) and lines[i].strip().startswith("*Table"):
            cap = lines[i].strip()
            i += 1
        add_table(rows[0], rows[2:], cap=cap)
        if cap and "Table 4" in cap:
            add_figure()
        continue
    # headings
    if s.startswith("# "):
        story.append(Paragraph(md_inline(s[2:]), S_TITLE)); i += 1; continue
    if s.startswith("## "):
        h = s[3:]
        if h.strip() == "References":
            in_ref = True
        story.append(Paragraph(md_inline(h), S_H1)); i += 1; continue
    if s.startswith("### "):
        story.append(Paragraph(md_inline(s[4:]), S_H2)); i += 1; continue
    # displayed equation?
    m = EQ_RE.search(s)
    if m and int(m.group(1)) in EQ_IMG:
        add_equation(int(m.group(1)))
        i += 1; continue
    # author block / abstract / keywords / body / references
    if not abstract_started:
        if s.startswith("**Abstract"):
            abstract_started = True
            story.append(Paragraph(md_inline(s), S_ABSTRACT))
        elif s.startswith("**Keywords"):
            story.append(Paragraph(md_inline(s), S_BODY))
        else:
            story.append(Paragraph(md_inline(s), S_AUTHOR))
        i += 1; continue
    if in_ref:
        story.append(Paragraph(md_inline(s), S_REF))
    else:
        story.append(Paragraph(md_inline(s), S_BODY))
    i += 1

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times", 8)
    canvas.drawCentredString(A4[0] / 2, 1.2 * cm, "Page %d" % doc.page)
    canvas.restoreState()

doc = SimpleDocTemplate(PDF, pagesize=A4,
                        leftMargin=3*cm, rightMargin=3*cm,
                        topMargin=3*cm, bottomMargin=3*cm,
                        title="Empirical Reproducibility Audit of MD-YOLO for Small-Object Detection in UAV Imagery",
                        author="Wang Qinmin")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("Saved PDF ->", PDF)
