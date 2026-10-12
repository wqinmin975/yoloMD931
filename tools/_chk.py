from pypdf import PdfReader
import re
r = PdfReader('MDYOLO_RSL_letter_submitted_v2.pdf')
txt = '\n'.join((p.extract_text() or '') for p in r.pages)
for ln in txt.split('\n'):
    if re.search(r'ORCID', ln):
        print(repr(ln))
print('Yuming ORCID present:', '0009-0003-7222-2390' in re.sub(r'\s+','',txt))
print('Qinmin ORCID still unique:', txt.count('0009-0006-0447-0482'))
print('PAGES:', len(r.pages))
