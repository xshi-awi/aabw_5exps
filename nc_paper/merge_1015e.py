#!/usr/bin/env python
"""
Make neutral density the reference coordinate for the thermal band, per
Xiaoxu's decision that the band should be quoted in gamma.

The two statements in the letter were not wrong, they were measured
differently. The summary quotes the band as drawn in Fig. R1.1a, classified
along the curve, gamma 27.10-27.68. The passages describing Fig. R2.3 quote
that figure's own sigma_2 bins, 36.75-37.00, which is gamma 27.33-27.59 and
sits inside the drawn band because the bins are coarser than the curve.

Fig. R2.3 is plotted against sigma_2, so its bin labels have to stay sigma_2
or the text would cite an axis the reader cannot see. The fix is to give the
gamma equivalent alongside each one, so a reader comparing the two figures
sees one band rather than two numbers that look inconsistent.
"""
import sys
from pathlib import Path

P = Path('RESPONSE_LETTER.md')
s = P.read_text()
edits = []


def sub(old, new, label):
    global s
    n = s.count(old)
    if n != 1:
        sys.exit('ANCHOR %r matched %d times, expected 1\n---\n%s\n---' % (label, n, old[:300]))
    s = s.replace(old, new)
    edits.append(label)


sub("""by density class, only one class in our pre-industrial state is thermally dominated, σ₂ =
36.75–37.00 kg m⁻³ at 70%, and every other class is haline dominated, falling to 6–7% thermal in
the densest classes.""",
    """by density class, only one class in our pre-industrial state is thermally dominated, σ₂ =
36.75–37.00 kg m⁻³, equivalently γ = 27.33–27.59 kg m⁻³, at 70%, and every other class is haline
dominated, falling to 6–7% thermal in the densest classes.""",
    'R2 Major issue 1: gamma equivalent for the thermal class')

sub("""state a single class, σ₂ = 36.75–37.00 kg m⁻³, reaches thermal dominance at 70%, while every
other class is haline dominated.""",
    """state a single class, σ₂ = 36.75–37.00 kg m⁻³, equivalently γ = 27.33–27.59 kg m⁻³, reaches
thermal dominance at 70%, while every other class is haline dominated.""",
    'Fig R2.3 caption: gamma equivalent')

sub("""σ₂ = 36.75–37.00 kg m⁻³ at 70% thermal. Every denser class is haline dominated, with sea ice
supplying 93% of the transformation above σ₂ = 37.00 kg m⁻³, and the lighter classes are dominated""",
    """σ₂ = 36.75–37.00 kg m⁻³, equivalently γ = 27.33–27.59 kg m⁻³, at 70% thermal. Every denser class
is haline dominated, with sea ice supplying 93% of the transformation above σ₂ = 37.00 kg m⁻³,
γ = 27.59 kg m⁻³, and the lighter classes are dominated""",
    'R3 density-range passage: gamma equivalents')

P.write_text(s)
print('applied %d edits:' % len(edits))
for e in edits:
    print('  -', e)
