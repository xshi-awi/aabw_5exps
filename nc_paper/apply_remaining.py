#!/usr/bin/env python
"""
The remaining yellow annotations:
  para 45  give a concrete example of the added lat/lon labels
  para 48  soften comment 1's bare "Corrected throughout"
  para 102 make R1 comment 26 consistent with the G1 answer, since both address
           the same observational comparison
  para 42  point the model-limitation reply at the MLD figure, which the author
           asked to be moved there rather than concentrated under G1
"""
import json
import re
import sys
from pathlib import Path

P = Path('RESPONSE_LETTER.md')
Q = json.loads(Path('quotes.json').read_text())
s = P.read_text()

JOBS = []


def E(anchor, replacement, label):
    JOBS.append((anchor, replacement, label))


# ---- para 45: concrete example of the labelling
E('We are happy to add complete labelling if the reviewer prefers, or to enlarge those figures to accommodate it.',
  'We are happy to add complete labelling if the reviewer prefers, or to enlarge those figures to '
  'accommodate it. As a concrete example, the ventilation age maps now carry labelled meridians at '
  '60°W, 0°, 60°E and 120°E as well as labelled latitude circles, and the mixed layer and composite '
  'figures carry labelled latitude circles; Figure R1.3 in this letter shows the result. The same '
  'treatment has been applied to every map figure in the revised manuscript and supplement.',
  'para45-example')

# ---- para 48: comment 1 deserves more than two words
E('**1.** *LIG, LGM and MIS should be capitalised.* — Corrected throughout.',
  '**1.** *LIG, LGM and MIS should be capitalised.* — Thank you for catching this. The three '
  'abbreviations are now capitalised consistently throughout the manuscript, the figure captions '
  'and the supplement, and MIS is written as MIS3 everywhere so that it matches the column headings '
  'of the figures.',
  'para48-thanks')

# ---- para 102: consistency between comment 26 and the G1 answer
E('The appendix figures are now also cited in\nnumerical order.',
  'The appendix figures are now also cited in numerical order. We note that this comment and the '
  'general comment on model evaluation above concern the same comparison, and we have kept the two '
  'answers consistent: the quantitative comparison with the published observational values is given '
  'once in the Discussion, and both replies point to the same new figure.',
  'para102-consistency')

for anchor, repl, label in JOBS:
    pat = re.compile(r'\s+'.join(re.escape(w) for w in anchor.split()))
    hits = list(pat.finditer(s))
    if len(hits) != 1:
        print(f'  NOT MATCHED {label}: {len(hits)}')
        continue
    s = s[:hits[0].start()] + repl + s[hits[0].end():]
    print(f'  applied {label}')

P.write_text(s)
print('done')
