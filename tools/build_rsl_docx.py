"""Build an RSL-compliant .docx from the RSL markdown (from scratch).

Times New Roman 11 pt, 3 cm margins, single spacing, tables, embedded
Figure 1, equations rendered as images with right-aligned numbers.
"""
import os, re
from docx import Document
from docx.shared import Cm, Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from PIL import Image as PILImage

MD = r"D:/ace/source/MDYOLO_RSL_letter.md"
FIG = r"D:/ace/source/RSL_fig1_domain_shift.png"
EQ_DIR = r"D:/ace/source/tools/_eq"
OUT_PRIMARY = r"D:/ace/source/MDYOLO_RSL_letter.docx"
OUT_FALLBACK = r"D:/ace/source/MDYOLO_RSL_letter_submitted.docx"

ACTIVE = {"n": 0}   # not using list numbering

# ---------- low-level helpers ----------
def set_run(run, size=None, bold=False, italic=False, mono=False):
    run.font.name = "Courier New" if mono else "Times New Roman"
    run.font.size = Pt(size if size else 11)
    run.bold = bold
    run.italic = italic
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts"); rpr.append(rf)
    rf.set(qn("w:ascii"), run.font.name)
    rf.set(qn("w:hAnsi"), run.font.name)
    rf.set(qn("w:eastAsia"), run.font.name)
    return run

TOKEN = re.compile(r"(\$[^$]+\$|\*\*.+?\*\*|\*[^*]+?\*|`[^`]+?`|\^[A-Za-z\u03b1-\u03c9\u0391-\u03a9])")

def add_runs(p, text, size=11, base_bold=False):
    """Inline formatting: **bold**, *italic*, `mono`, ^super."""
    for part in TOKEN.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            set_run(p.add_run(part[2:-2]), size, bold=True)
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            set_run(p.add_run(part[1:-1]), size, italic=True)
        elif part.startswith("`") and part.endswith("`") and len(part) > 2:
            set_run(p.add_run(part[1:-1]), size, mono=True)
        elif part.startswith("$") and part.endswith("$") and len(part) > 2:
            inner = re.sub(r"\\text\{([^}]*)\}", r"\1", part[1:-1])
            set_run(p.add_run(inner), size)
        elif part.startswith("^") and len(part) == 2:
            r = set_run(p.add_run(part[1]), size, bold=base_bold)
            r.font.superscript = True
        else:
            set_run(p.add_run(part), size, bold=base_bold)

def body_para(doc, text, size=11, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
              space_after=6, hanging=False, center=False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if center else align
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.0
    if hanging:
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.first_line_indent = Inches(-0.25)
    add_runs(p, text, size=size)
    return p

def heading(doc, text, size, space_before=10, space_after=4):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.0
    set_run(p.add_run(text), size=size, bold=True)
    return p

def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear"); sh.set(qn("w:fill"), hexcolor)
    tcPr.append(sh)

def add_table(doc, header, body):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(header):
        c = t.rows[0].cells[j]
        c.paragraphs[0].paragraph_format.space_after = Pt(0)
        add_runs(c.paragraphs[0], h, size=9, base_bold=True)
        shade(c, "E8EEF5")
    for row in body:
        cells = t.add_row().cells
        for j, val in enumerate(row):
            cells[j].paragraphs[0].paragraph_format.space_after = Pt(0)
            add_runs(cells[j].paragraphs[0], val, size=9)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t

EQ_RE = re.compile(r"\(\s*(\d+)\s*\)\s*$")

def add_equation(doc, n):
    path = os.path.join(EQ_DIR, f"eq{n}.png")
    if not os.path.exists(path):
        body_para(doc, "Equation (%d)" % n, center=True); return
    w, h = PILImage.open(path).size
    # insert at natural size (rendered at 200 dpi); only cap if wider than the cell
    nat_w_in = w / 200.0
    max_w_in = 4.7   # equation cell width, inches
    target_w_in = min(nat_w_in, max_w_in)
    target_w = Inches(target_w_in)
    target_h = Inches(target_w_in * h / w)
    t = doc.add_table(rows=1, cols=2)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    # borderless
    tblPr = t._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}"); e.set(qn("w:val"), "nil"); borders.append(e)
    tblPr.append(borders)
    c0, c1 = t.rows[0].cells
    p0 = c0.paragraphs[0]; p0.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p0.add_run().add_picture(path, width=target_w, height=target_h)
    p1 = c1.paragraphs[0]; p1.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    set_run(p1.add_run("(%d)" % n), size=11)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)

def add_figure(doc):
    if not os.path.exists(FIG):
        return
    w, h = PILImage.open(FIG).size
    iw = Inches(6.0)
    ih = Inches(6.0 * h / w)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(FIG, width=iw, height=ih)
    cap = doc.add_paragraph(); cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(8)
    set_run(cap.add_run(
        "Figure 1. Cross-domain localization degradation from VisDrone "
        "(in-domain) to UAVDT-M (zero-shot). (a) Mean best per-image IoU "
        "drops from 0.40 to 0.27. (b) Fraction of test images achieving at "
        "least 0.25 and 0.50 IoU overlap, showing performance degradation "
        "under domain shift."),
        size=9, italic=True)

# ---------- build ----------
doc = Document()
for sec in doc.sections:
    sec.left_margin = sec.right_margin = sec.top_margin = sec.bottom_margin = Cm(3)

normal = doc.styles["Normal"]
normal.font.name = "Times New Roman"
normal.font.size = Pt(11)
normal.paragraph_format.line_spacing = 1.0
normal.paragraph_format.space_after = Pt(6)
normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")

with open(MD, encoding="utf-8") as f:
    lines = f.read().replace("\u2212", "-").split("\n")

abstract_started = False
in_ref = False
i = 0
while i < len(lines):
    s = lines[i].strip()
    if s == "" or s in ("---", "***"):
        i += 1; continue
    if s.startswith("```"):
        i += 1; buf = []
        while i < len(lines) and not lines[i].strip().startswith("```"):
            buf.append(lines[i]); i += 1
        i += 1
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.2)
        set_run(p.add_run("\n".join(buf)), size=9, mono=True)
        continue
    if (s.startswith("|") and i + 1 < len(lines)
            and lines[i + 1].strip().startswith("|")
            and set(lines[i + 1].strip().replace(" ", "")) <= set("|-:")):
        tbl = []
        j = i
        while j < len(lines) and lines[j].strip().startswith("|"):
            tbl.append(lines[j].strip()); j += 1
        rows = [[c.strip() for c in r.strip("|").split("|")] for r in tbl]
        add_table(doc, rows[0], rows[2:])
        i = j
        while i < len(lines) and lines[i].strip() == "":
            i += 1
        if i < len(lines) and lines[i].strip().startswith("*Table"):
            cap = lines[i].strip()
            body_para(doc, cap, size=9, center=True, space_after=8)
            i += 1
            if "Table 4" in cap:
                add_figure(doc)
        continue
    if s.startswith("# "):
        body_para(doc, s[2:], size=16, center=True, space_after=4)
        i += 1; continue
    if s.startswith("## "):
        h = s[3:]
        if h.strip() == "References":
            in_ref = True
        heading(doc, h, size=13, space_before=12, space_after=5)
        i += 1; continue
    if s.startswith("### "):
        heading(doc, s[4:], size=11.5, space_before=8, space_after=4)
        i += 1; continue
    m = EQ_RE.search(s)
    if m and os.path.exists(os.path.join(EQ_DIR, f"eq{m.group(1)}.png")):
        add_equation(doc, int(m.group(1)))
        i += 1; continue
    if not abstract_started:
        if s.startswith("**Abstract"):
            abstract_started = True
            body_para(doc, s, size=11, space_after=6)
        elif s.startswith("**Keywords"):
            body_para(doc, s, size=11, space_after=8)
        else:
            body_para(doc, s, size=11, center=True, space_after=2)
        i += 1; continue
    if in_ref:
        body_para(doc, s, size=10, hanging=True, space_after=3)
    else:
        body_para(doc, s, size=11)
    i += 1

out = OUT_PRIMARY
try:
    doc.save(out)
except PermissionError:
    out = OUT_FALLBACK
    doc.save(out)
print("Saved DOCX ->", out)
