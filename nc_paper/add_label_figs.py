#!/usr/bin/env python
"""
Two annotations asked for a figure that was never supplied:

  G5 (lat/lon labels)  give a concrete example of the labelling
  comment 17 (Fig. 3)  show the figure with the new climate-state column headings

Both are added here, and then every figure label in the letter is renumbered from
scratch in document order, per reviewer, so that R1.x, R2.x and R3.x each run
1..n with no duplicates.
"""
import re
import sys
from pathlib import Path

P = Path('RESPONSE_LETTER.md')
s = P.read_text()

IMG = 'letter_figs'
added = 0

# ---------------------------------------------------------------- G5 example
anchor_g5 = ('The same treatment has been applied to every map figure in the revised manuscript '
             'and supplement.')
pat = re.compile(r'\s+'.join(re.escape(w) for w in anchor_g5.split()))
h = list(pat.finditer(s))
if len(h) == 1:
    block = f"""

![]({IMG}/fig_age_labels-1.png)

**Figure RX_AGE (revised manuscript figure).** The ventilation age figure as an example of the
labelling now applied throughout. Every panel carries labelled latitude circles and labelled
meridians, and the climate state is named above each panel. The mixed layer, wind stress and
composite figures have been treated the same way; please refer to the revised manuscript and
supplement for the complete set."""
    s = s[:h[0].end()] + block + s[h[0].end():]
    added += 1
else:
    print('  G5 anchor:', len(h))

# ---------------------------------------------------------------- comment 17
anchor_17 = 'Added, here and on the other multi-panel figures; Reviewer 3 made the same request.'
pat = re.compile(r'\s+'.join(re.escape(w) for w in anchor_17.split()))
h = list(pat.finditer(s))
if len(h) == 1:
    block = f"""

![]({IMG}/fig03_labels-1.png)

**Figure RX_F3 (revised manuscript Fig. 3).** The surface density tendency figure with the
climate-state headings added, so that the columns are identified in the same way as in the other
figures. The freshwater counterpart, Fig. 4, has been given the same headings."""
    s = s[:h[0].end()] + block + s[h[0].end():]
    added += 1
else:
    print('  comment 17 anchor:', len(h))

# ---------------------------------------------------------------- renumber
SECTIONS = [('R1', '# Reviewer #1', '# Reviewer #2'),
            ('R2', '# Reviewer #2', '# Reviewer #3'),
            ('R3', '# Reviewer #3', '# Reviewer #4')]

for tag, a_mark, b_mark in SECTIONS:
    a = s.find(a_mark)
    b = s.find(b_mark)
    if a < 0 or b < 0:
        continue
    sec = s[a:b]
    labels = re.findall(r'\*\*Figure ([A-Za-z0-9_.]+)\b', sec)
    mapping, i = {}, 0
    for old in labels:
        i += 1
        mapping[old] = f'{tag}.{i}'
    # placeholder pass so renames cannot collide
    for old, new in mapping.items():
        sec = sec.replace(f'**Figure {old} ', f'**Figure @@{new}@@ ')
        sec = sec.replace(f'Figure {old} above', f'Figure @@{new}@@ above')
        sec = sec.replace(f'Figure {old}.', f'Figure @@{new}@@.')
        sec = sec.replace(f'Figure {old},', f'Figure @@{new}@@,')
    sec = sec.replace('@@', '')
    s = s[:a] + sec + s[b:]

P.write_text(s)
print(f'added {added} figures, renumbered all sections')

# ---------------------------------------------------------------- verify
s = P.read_text()
ok = True
for tag, a_mark, b_mark in SECTIONS:
    sec = s[s.find(a_mark):s.find(b_mark)]
    labs = re.findall(r'\*\*Figure ([A-Za-z0-9_.]+)\b', sec)
    want = [f'{tag}.{i+1}' for i in range(len(labs))]
    status = 'OK' if labs == want else 'MISMATCH'
    if labs != want:
        ok = False
    print(f'  {tag}: {labs}  {status}')

refs = set(re.findall(r'Figure (R[0-9]\.[0-9]+)', s))
defs = set(re.findall(r'\*\*Figure (R[0-9]\.[0-9]+)', s))
dangling = sorted(refs - defs)
print('  dangling references:', dangling or 'none')
sys.exit(0 if ok and not dangling else 1)
