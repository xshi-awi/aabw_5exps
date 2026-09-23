#!/usr/bin/env python
"""
Consistency checks for the response letter. Run after any edit.

  1 every figure label is sequential within its reviewer section
  2 every in-prose cross-reference points at a label that exists
  3 the number of labels equals the number of embedded images
  4 no reviewer comment is elided with an ellipsis
  5 every italic quotation is traceable to quotes.json
"""
import json
import re
import sys
from pathlib import Path

s = Path('RESPONSE_LETTER.md').read_text()
Q = json.loads(Path('quotes.json').read_text()) if Path('quotes.json').exists() else {}
fail = []

SEC = [('R1', '# Reviewer #1', '# Reviewer #2'),
       ('R2', '# Reviewer #2', '# Reviewer #3'),
       ('R3', '# Reviewer #3', '# Reviewer #4')]

for tag, a, b in SEC:
    sec = s[s.find(a):s.find(b)]
    labs = re.findall(r'\*\*Figure ([A-Za-z0-9_.]+)\b', sec)
    imgs = re.findall(r'!\[\]\(letter_figs/', sec)
    want = [f'{tag}.{i+1}' for i in range(len(labs))]
    if labs != want:
        fail.append(f'{tag} labels not sequential: {labs}')
    if len(labs) != len(imgs):
        fail.append(f'{tag} has {len(labs)} labels but {len(imgs)} images')
    print(f'  {tag}: {len(labs)} figures {labs}')

defs = set(re.findall(r'\*\*Figure (R[0-9]\.[0-9]+)', s))
refs = set(re.findall(r'Figure (R[0-9]\.[0-9]+)', s))
dang = sorted(refs - defs)
if dang:
    fail.append(f'dangling cross-references: {dang}')

ell = len(re.findall(r'^>.*\.\.\.', s, re.M))
if ell:
    fail.append(f'{ell} reviewer comment lines still contain an ellipsis')

quotes = re.findall(r'> \*"(.*?)"\*', s, re.S)
orphan = [q for q in quotes if not any(v.strip() == q.strip() for v in Q.values())]
if orphan:
    fail.append(f'{len(orphan)} quotations not traceable to quotes.json')

print(f'  cross-references: {len(refs)} used, {len(dang)} dangling')
print(f'  quotations: {len(quotes)}, {len(orphan)} untraceable')
print(f'  elided comments: {ell}')

# ---------------------------------------------------------------- deliverables
# The markdown being right is not enough. A caption that never reaches the
# LaTeX or the .docx is invisible to the reviewer, which is exactly how the
# \caption* mismatch slipped through once already.
n_md = len(re.findall(r'\*\*Figure R[0-9]\.[0-9]+', s))

tex = Path('SUBMISSION/04_response_letter.tex')
if tex.exists():
    t = tex.read_text()
    n_inc = t.count('\\includegraphics')
    n_cap = len(re.findall(r'\\textbf\{Figure R[0-9]\.[0-9]+', t))
    print(f'  tex: {n_inc} images, {n_cap} captions')
    if n_cap != n_md or n_inc != n_md:
        fail.append(f'tex has {n_inc} images / {n_cap} captions, markdown has {n_md}')

docx = Path('SUBMISSION/05_response_letter.docx')
if docx.exists():
    import zipfile
    with zipfile.ZipFile(docx) as z:
        x = z.read('word/document.xml').decode('utf-8')
    n_draw = x.count('<w:drawing>')
    n_dcap = len(set(re.findall(r'Figure (R[0-9]\.[0-9]+)', x)))
    print(f'  docx: {n_draw} images, {n_dcap} distinct figure labels')
    if n_draw != n_md:
        fail.append(f'docx has {n_draw} images, markdown has {n_md}')
    if n_dcap != n_md:
        fail.append(f'docx names {n_dcap} distinct figures, markdown defines {n_md}')

if fail:
    print('\nFAIL')
    for f in fail:
        print('  -', f)
    sys.exit(1)
print('\nall checks pass')
