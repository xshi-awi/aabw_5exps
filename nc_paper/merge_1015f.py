#!/usr/bin/env python
"""
Say in the Fig. R1.1 caption how the shaded band is classified, and give the
band limits, so the reader can reconcile it with the coarser sigma_2 bins
quoted for Fig. R2.3.
"""
import sys
from pathlib import Path

OLD = """Pink shading in (a) marks where the heat term carries the total and blue shading where the
freshwater term does, classified along the drawn curve."""

NEW = """Pink shading in (a) marks where the heat term carries the total and blue shading where the
freshwater term does, classified point by point along the drawn curve rather than by density bin,
which places the thermally driven band at γ = 27.10–27.68 kg m⁻³. Where the same band is quoted
from the binned diagnostic of Fig. R2.3 it appears as γ = 27.33–27.59 kg m⁻³, the single bin that
falls inside it."""

OLD_R1 = """Pink shading in (a) marks where the heat term carries the total and blue shading where the
freshwater term does, classified along the drawn curve."""

NEW_R1 = """Pink shading in (a) marks where the heat term carries the total and blue shading where the
freshwater term does, classified point by point along the drawn curve rather than by density bin,
which places the thermally driven band at γ = 27.10–27.68 kg m⁻³."""

for f, old, new in (('RESPONSE_LETTER.md', OLD, NEW),
                    ('RESPONSE_LETTER_R1.md', OLD_R1, NEW_R1)):
    P = Path(f)
    s = P.read_text()
    n = s.count(old)
    if n != 1:
        sys.exit('%s: anchor matched %d times, expected 1' % (f, n))
    P.write_text(s.replace(old, new))
    print('patched', f)
