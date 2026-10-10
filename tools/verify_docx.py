from docx import Document
import sys
path = sys.argv[1]
d = Document(path)
print("paragraphs:", len(d.paragraphs))
print("tables:", len(d.tables))
print("inline images:", len(d.inline_shapes))
body = "\n".join(p.text for p in d.paragraphs)
cjk = [c for c in body if '\u4e00' <= c <= '\u9fff']
print("CJK chars:", len(cjk))
for cap in ["Table 2.", "Table 3.", "Table 4."]:
    print(f"  caption '{cap}':", body.count(cap))
# table shapes: first cell of each table
print("--- table headers ---")
for ti, t in enumerate(d.tables):
    try:
        print(f"  table {ti}: cols={len(t.columns)} rows={len(t.rows)} | header={[c.text for c in t.rows[0].cells]}")
    except Exception as e:
        print("  table", ti, "err", e)
print("--- headings (bold >= 11.5) ---")
for p in d.paragraphs:
    if p.runs and p.runs[0].bold and (p.runs[0].font.size and p.runs[0].font.size.pt >= 11.5):
        print("  ", p.text[:70])
print("loU present?", "loU" in body, "| IoU present?", "IoU" in body)
print("U+FFEB present?", "\ufeeb" in body)
