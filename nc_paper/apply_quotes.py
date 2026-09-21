#!/usr/bin/env python
"""
Add italic quotations of the revised manuscript text to the response letter.

The author's annotation asks that every substantive reply quote the actual new
wording, so a reviewer reading only the letter sees exactly what changed.
Quotations come from quotes.json, extracted verbatim from build/revised.tex, so
they cannot drift from the manuscript.

Anchors are short unique fragments rather than whole paragraphs, because the
letter's line wrapping does not match hand-typed text.
"""
import json
import re
import sys
from pathlib import Path

P = Path('RESPONSE_LETTER.md')
Q = json.loads(Path('quotes.json').read_text())
s = P.read_text()

LEAD_DEFAULT = 'In the revised manuscript this now reads:'

# (anchor fragment, quote key, lead-in, label)
JOBS = [
    ('governs the response to internal atmospheric\nvariability.', 'abstract_sam', None, 'r1-2'),
    ('before the ages are given. This also incorporates the new overturning\nanalysis requested by Reviewer 3.', 'abstract_vent', None, 'r1-4'),
    ('cites the observational literature linking it to\nbottom water changes.', 'intro_sam', None, 'r1-5'),
    ('we now quantify the reduction rather than describing it qualitatively.', 'mld_quant', None, 'r1-9'),
    ('including the sign convention and the physical meaning of brine rejection versus melt.', 'tendency_defn', None, 'r1-19'),
    ('and that in the glacial states the signal is carried largely by the Weddell sector.', 'sam_nonuniform', None, 'r1-28'),
    ('a practical route to a marine reconstruction.', 'sediment', None, 'r1-29'),
    ('longwave fluxes and the turbulent component the sum of the latent and sensible heat fluxes.', 'components', None, 'r1-30'),
    ('so that pathway is absent from our simulations.', 'meltwater', None, 'r3-meltwater'),
]

added, missed = 0, []
for anchor, key, lead, label in JOBS:
    # whitespace-insensitive match, so line wrapping does not matter
    pat = re.compile(r'\s+'.join(re.escape(w) for w in anchor.split()))
    hits = list(pat.finditer(s))
    if len(hits) != 1:
        missed.append((label, len(hits)))
        continue
    end = hits[0].end()
    block = ('\n\n' + (lead or LEAD_DEFAULT) + '\n\n> *"' + Q[key] + '"*')
    s = s[:end] + block + s[end:]
    added += 1

# --- replies that need the quotation inserted by locating the numbered comment
def after_reply(num, key, lead=None):
    """Append a quotation to the reply that follows comment `num`."""
    global s, added, missed
    m = re.search(r'\n\*\*%s\.\*\* \*[^\n]*' % num, s)
    if not m:
        missed.append((f'comment {num}', 0))
        return
    # the reply runs from the end of the italic comment to the next ** marker
    start = m.end()
    nxt = s.find('\n**', start)
    if nxt < 0:
        missed.append((f'comment {num} end', 0))
        return
    block = ('\n\n' + (lead or LEAD_DEFAULT) + '\n\n> *"' + Q[key] + '"*')
    s = s[:nxt] + block + s[nxt:]
    added += 1


for num, key in [('11', 'density_panels'), ('14', 'thermal_quant'), ('27', 'glacial_quant')]:
    after_reply(num, key)

P.write_text(s)
print(f'added {added} italic quotations')
if missed:
    print('not matched:')
    for label, n in missed:
        print(f'  {label}: {n}')
