import pymupdf, sys
path = sys.argv[1]
doc = pymupdf.open(path)
print("PDF pages:", doc.page_count)
for n in range(doc.page_count):
    imgs = doc[n].get_images(full=True)
    txt = doc[n].get_text()
    print(f"  page {n+1}: images={len(imgs)} | 'Figure 1'={'Figure 1' in txt} | has 'Table'={('Table' in txt)}")
full = "\n".join(doc[n].get_text() for n in range(doc.page_count))
for cap in ["Table 2.", "Table 3.", "Table 4.", "Table S1"]:
    print(f"  caption '{cap}' count:", full.count(cap))
# count equation images: eq png xobjects
eqcount = 0
for n in range(doc.page_count):
    for im in doc[n].get_images(full=True):
        xref = im[0]
        try:
            if "eq" in (doc.extract_image(xref).get("filename","").lower()):
                eqcount += 1
        except Exception:
            pass
print("  equation png xobjects:", eqcount)
print("  'loU' present?", "loU" in full, "| 'IoU' present?", "IoU" in full)
print("  U+FFEB present?", "\ufeeb" in full)
print("  abstract words (approx):", len([w for w in full.split() if w.startswith(('Abstract','We','Previous','real','four','After','Nevertheless','Diagnostic','Further','zero-shot','UAVDT')) or w[0].islower()]) if False else "see below")
