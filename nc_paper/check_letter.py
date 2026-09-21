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

if fail:
    print('\nFAIL')
    for f in fail:
        print('  -', f)
    sys.exit(1)
print('\nall checks pass')
