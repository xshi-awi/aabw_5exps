#!/usr/bin/env python
"""
Drop Figs R3.1, R3.2 and R3.3 and the text that carries them, per Xiaoxu:
the first Reviewer 3 comment needs only the comparison figure.

What goes, and what has to survive the cut:

R3.1, the mixed layer maps, illustrated the diagnosis that the model convects
in the open Weddell gyre. The figure goes, but the finding itself is the
concession the reviewer asked for, so the numbers stay in the prose. The same
map is still shown to Reviewer 2 as Fig R2.4, so nothing is lost from the
letter as a whole.

R3.2 and R3.3 showed the per-class decomposition and the domain sensitivity.
Both go. The conclusions they supported, the thermal band and the fact that
restricting the domain does not reverse the partition, stay as prose, and the
merged comparison figure now carries the visual argument on its own.

One cross-reference has to be rewritten: the ORAS5 paragraph attributed the
winter amplitude gap to "the bias diagnosed above in Fig. R3.1", which would
dangle once that figure is gone.
"""
import sys
from pathlib import Path

P = Path('nc_paper/RESPONSE_LETTER.md')
s = P.read_text()
edits = []


def sub(old, new, label):
    global s
    n = s.count(old)
    if n != 1:
        sys.exit('ANCHOR %r matched %d times, expected 1\n---\n%s\n---' % (label, n, old[:260]))
    s = s.replace(old, new)
    edits.append(label)


# ---- R3.1: drop the figure, keep the diagnosis in prose
sub("""![](letter_figs/figR15_mld_maps_5exps.png)

**Figure R3.1 (new analysis).** Winter mixed layer depth south of 60°S for the five states. (a-c) use a 0-800 m scale with the 400 m contour in orange, (d-e) a 0-300 m scale with the 200 m
contour, because the glacial mixed layer never approaches 400 m and would be featureless on the
shared scale.
The pre-industrial deep mixed layers lie in the interior of the Weddell gyre rather than on the
shelf, which is the bias the reviewer identified, and the same holds in MH and LIG. In the glacial
states the deep convection contracts to narrow coastal cells.

The model therefore does form its dense water by open-ocean convection, as""",
    """The same holds in MH and LIG, while in the glacial states the deep convection contracts to narrow
coastal cells. The model therefore does form its dense water by open-ocean convection, as""",
    'R3.1 figure removed, diagnosis kept as prose')

# ---- R3.2 and R3.3: drop both figures and their captions
sub("""![](letter_figs/figR14_siz_gamma_5exps.png)

**Figure R3.2 (new analysis).** Transformation in the seasonal sea ice zone, resolved by density
class, for all five climate states. The upper row gives the total transformation in grey with its
thermal, sea ice and other freshwater contributions; the lower row gives the thermal share, with
the haline dominated band shaded green and the thermally dominated band shaded pink. Only one
pre-industrial class reaches thermal dominance. Both glacial states are haline dominated in every
class, at 4-28% thermal, which is the regime shift the paper reports. Because glacial surface water
is denser, the class boundaries for LGM and MIS3 are shifted by +0.80 and +0.50 kg m⁻³, the density
anomaly produced at the freezing point by the +1.0 and +0.6 salinity increase prescribed in those
experiments.

![](letter_figs/figR1_wmt_domain_sensitivity.png)

**Figure R3.3 (manuscript Fig. 8).** The like-for-like comparison the reviewer asked for.
Transformation integrated over the published domain south of 60°S (top) and over the seasonal sea
ice zone that reproduces the observational domain (bottom). The haline share rises in the
restricted domain, but the glacial-interglacial contrast is unaffected, which is why we conclude
the regime shift is not an artefact of the biased pre-industrial end member.

We also took the second suggestion,""",
    """Both comparisons are shown in the manuscript as Fig. 8, where the transformation is integrated
over the published domain south of 60°S and over the seasonal sea ice zone that reproduces the
observational domain.

We also took the second suggestion,""",
    'R3.2 and R3.3 figures removed, pointer to manuscript Fig. 8 kept')

# ---- the cross-reference that would dangle
sub("""more transformation, and that bias is the same one diagnosed above in Fig. R3.1.""",
    """more transformation, and that bias is the same open-ocean convection diagnosed above.""",
    'ORAS5 cross-reference to R3.1 rewritten')

P.write_text(s)
print('applied %d edits:' % len(edits))
for e in edits:
    print('  -', e)
