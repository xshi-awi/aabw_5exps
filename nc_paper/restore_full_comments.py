#!/usr/bin/env python
"""
Replace every abbreviated reviewer comment in the response letter with the
complete verbatim text from comments.txt.

The author's point is that a reviewer may not remember their own wording, so
every comment must be reproduced in full rather than elided with "...".

Each entry gives the first and last few words of the comment as it appears in
comments.txt; the script pulls everything between them, wraps it as a markdown
blockquote in italics, and swaps it for whatever truncated version is currently
in the letter (located by its own opening words).
"""
import re
import sys
from pathlib import Path

SRC = Path('comments.txt')
LET = Path('RESPONSE_LETTER.md')

src = SRC.read_text()
src = src[src.find('Reviewer #1 (Remarks to the Author):'):]
letter = LET.read_text()


def full(start, end):
    """Verbatim comment text from comments.txt, between the two markers."""
    i = src.find(start)
    if i < 0:
        return None
    j = src.find(end, i)
    if j < 0:
        return None
    t = src[i:j + len(end)]
    t = re.sub(r'\s*\n\s*', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()


def quote(t):
    """Wrap as a markdown blockquote in italics, wrapped at a sane width."""
    words = t.split()
    lines, cur = [], '> *'
    for w in words:
        if len(cur) + len(w) + 1 > 96:
            lines.append(cur)
            cur = '> ' + w
        else:
            cur = cur + (' ' if cur not in ('> *',) else '') + w
    lines.append(cur + '*')
    return '\n'.join(lines)


# (marker in the LETTER identifying the block to replace, start, end in comments.txt)
JOBS = [
    ('> *The manuscript presents a comprehensive investigation',
     'The manuscript presents a comprehensive investigation',
     'improve the clarity of the presentation.'),

    ('> *The conclusions of this study rely entirely',
     'The conclusions of this study rely entirely',
     'substantially increase confidence in the conclusions.'),

    ('> *The description of the prescribed boundary conditions',
     'The description of the prescribed boundary conditions',
     'easier to follow.'),

    ('> *Please consider adding latitude and longitude labels',
     'Please consider adding latitude and longitude labels',
     'make the discussion easier to follow.'),

    ('> *There is a mismatch with observational studies',
     '1) There is a mismatch with observational studies',
     'this seems like a major problem.'),

    ("> *I'm puzzled that they don't acknowledge",
     "2) I’m puzzled that they don’t acknowledge",
     'already discussed in that paper?'),

    ('> *The fundamental issue must be getting right',
     'The fundamental issue must be getting right',
     'would add to the value of the paper.'),

    ('> *The authors highlight the lack of deep convection',
     'The authors highlight the lack of deep convection',
     'how most PMIP models fail in this regard.)'),

    ('> *Shi et al apply a water mass transformation',
     'Shi et al apply a water mass transformation',
     'interesting stabilising mechanism for ventilation.'),

    ('> *... to the best of our knowledge today densewater',
     'While this is an interesting observation in model space',
     'overlooked process in the real modern ocean*.'),

    ('> *how applicable is this result to thinking about',
     'While the glacial-interglacial shift in regime',
     'I do not believe this is not the case in the current framing.'),

    ('> *The authors mention that they use ideal age',
     'The authors mention that they use ideal age',
     'to equilibrate (see Millet et al. 2025).'),

    ('> *L25: change deep from',
     'L25: change deep from',
     'assess the overall impact on ventilation.'),
]

done, missed = 0, []
for marker, a, b in JOBS:
    new = full(a, b)
    if not new:
        missed.append(('source', marker[:40]))
        continue
    i = letter.find(marker)
    if i < 0:
        missed.append(('letter', marker[:40]))
        continue
    # the blockquote runs until the first line that is not part of it
    j = i
    while j < len(letter):
        nl = letter.find('\n', j)
        if nl < 0:
            nl = len(letter)
        line = letter[j:nl]
        if not line.startswith('>') and line.strip() != '':
            break
        if not line.startswith('>') and line.strip() == '':
            # blank line ends the quote unless the next line continues it
            nxt = letter.find('\n', nl + 1)
            follow = letter[nl + 1:nxt if nxt > 0 else len(letter)]
            if not follow.startswith('>'):
                break
        j = nl + 1
    letter = letter[:i] + quote(new) + '\n' + letter[j:]
    done += 1

LET.write_text(letter)
print(f'restored {done} comments in full')
for kind, m in missed:
    print(f'  MISSED ({kind}): {m}')
