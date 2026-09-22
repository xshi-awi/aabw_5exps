#!/usr/bin/env python
"""
Add the density-class figure to the response letter, and correct the
"69% to 61%" claim that contradicts it.

The claim came from integrating over sigma2 >= 36.6-36.8, which is neither a
dense class nor a boundary with any justification behind it. Measured at the
transformation maximum the thermal share does not fall when the domain is
restricted, it rises slightly, 68% to 70%, because the heat and sea ice terms
weaken together. The figure that accompanies the claim shows those two numbers,
so the letter contradicted its own figure.

What the new figure shows is stronger and survives checking: within the ice
zone only one density class in PI is thermally dominated, everything else is
haline dominated, and both glacial states are haline dominated throughout.
"""
import re
import sys
from pathlib import Path

P = Path('RESPONSE_LETTER.md')
s = P.read_text()

IMG = 'letter_figs/figR14_siz_gamma_5exps.png'

# ---------------------------------------------------------------- the block
BLOCK = """

![](%s)

**Figure R2.3 (new analysis).** Transformation in the seasonal sea ice zone, the
Pellichero-comparable domain, resolved by density class. The upper row separates the heat, sea ice
and other freshwater contributions; the lower row gives the thermal share, with the haline
dominated band shaded green and the thermally dominated band shaded red. In the pre-industrial
state a single class, σ₂ = 36.75–37.00 kg m⁻³, reaches thermal dominance at 70%%, while every
other class is haline dominated. Both glacial states are haline dominated in every class, with
thermal shares between 4%% and 28%%. Because glacial surface water is denser, the class boundaries
for LGM and MIS3 are shifted by +0.80 and +0.50 kg m⁻³, the density anomaly produced at the
freezing point by the +1.0 and +0.6 salinity increase prescribed in those experiments.""" % IMG

# ---------------------------------------------------------------- correction
OLD = ("Recomputing our transformation over a sea ice sector defined exactly as theirs, "
       "which reproduces their domain to within 2.5% in area, moves the partition in the direction "
       "they report: the thermal share of the dense-class transformation falls from 69% to 61%, "
       "and the sea ice contribution at the transformation maximum roughly doubles from 0.86 to "
       "1.84 Sv.")

NEW = ("Recomputing our transformation over a sea ice sector defined exactly as theirs, which "
       "reproduces their domain to within 2.5% in area, sharpens the picture considerably. "
       "Resolved by density class, only one class in our pre-industrial state is thermally "
       "dominated, σ₂ = 36.75–37.00 kg m⁻³ at 70%, and every other class is haline dominated, "
       "falling to 6–7% thermal in the densest classes. Both glacial states are haline dominated "
       "in every class, between 4% and 28%. The regime shift the paper reports is therefore not "
       "only preserved inside the observational domain, it is clearer there than in the wider "
       "domain we originally used.")

hits = s.count(OLD)
if hits != 1:
    # the letter wraps lines, so match whitespace-insensitively
    pat = re.compile(r'\s+'.join(re.escape(w) for w in OLD.split()))
    m = list(pat.finditer(s))
    if len(m) != 1:
        print('correction anchor matched %d times' % len(m))
        sys.exit(1)
    s = s[:m[0].start()] + NEW + s[m[0].end():]
else:
    s = s.replace(OLD, NEW)
print('corrected the 69/61 claim')

# ---------------------------------------------------------------- bias paragraph
BIAS_ANCHOR = ("We now state in the Results that the interglacial thermal pathway we identify is "
               "more relevant to intermediate and mode water formation than to bottom water proper.")
BIAS_ADD = (" We should be equally direct about what this comparison does not resolve. Our "
            "pre-industrial Southern Ocean carries real biases: the surface water around "
            "Antarctica is too warm and too fresh, the model convects in the open Weddell gyre "
            "rather than over the shelf, and it consequently ventilates the densest classes far "
            "too weakly. Those biases are why our absolute transformation rates in the bottom "
            "water classes are small. They do not, however, undermine the result the paper "
            "reports, because that result is a comparison between climate states computed with "
            "one model, one domain definition and one diagnostic. The bias is common to all five "
            "experiments and therefore largely divides out of the differences between them. The "
            "shift from thermally influenced interglacials to strongly haline glacials is visible "
            "in every density class and in both domains we have tested, which is the sense in "
            "which we consider it robust.")

pat = re.compile(r'\s+'.join(re.escape(w) for w in BIAS_ANCHOR.split()))
m = list(pat.finditer(s))
if len(m) != 1:
    print('bias anchor matched %d times' % len(m))
    sys.exit(1)
s = s[:m[0].end()] + BIAS_ADD + s[m[0].end():]
print('added the bias paragraph')

# ---------------------------------------------------------------- insert figure
CAP_ANCHOR = ("thermally dominated interglacials and haline dominated glacials survives intact.")
pat = re.compile(r'\s+'.join(re.escape(w) for w in CAP_ANCHOR.split()))
m = list(pat.finditer(s))
if len(m) != 1:
    print('figure anchor matched %d times' % len(m))
    sys.exit(1)
s = s[:m[0].end()] + BLOCK + s[m[0].end():]
print('inserted the figure')

P.write_text(s)
print('done')
