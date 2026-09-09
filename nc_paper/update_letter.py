#!/usr/bin/env python
"""
Update the response letter:
  - embed the figures so reviewers do not have to consult the manuscript
  - Reviewer 2: state that the SAM figures have moved to the supplement
  - Reviewer 3: present the strengthened SAM analysis and its new results
"""
import sys
from pathlib import Path

P = Path('RESPONSE_LETTER.md')
s = P.read_text()
edits = []


def E(old, new, label):
    edits.append((old, new, label))


# ------------------------------------------------------------------ preamble
E("""We have also added the analyses that Reviewer 3 requested, showing the overturning streamfunction
directly and ideal age throughout the global ocean; discussed the study by Chen et al. (2025) that
Reviewer 2 correctly noted was missing; and worked through every specific comment. Three new
figures and one new supplementary figure have been added, along with a boundary condition table.""",
"""We have also added the analyses that Reviewer 3 requested, showing the overturning streamfunction
directly, ideal age throughout the global ocean, and a fuller treatment of the Southern Annular
Mode; discussed the study by Chen et al. (2025) that Reviewer 2 correctly noted was missing; and
worked through every specific comment. Five new figures have been added, along with a boundary
condition table and a Source Data file.

For convenience the principal new figures are reproduced in this letter, so that the reviewers can
assess them without consulting the manuscript.""",
  'preamble-figs')

# ------------------------------------------------------------------ R1 G1 figure
E("""comparison is now like-for-like. The result is shown in the new Fig. 8 and discussed in the
Discussion. Restricting the integration to the ice-covered sector raises the haline share, as
expected, moving the thermal fraction of the dense-class transformation from 69% to 61% and roughly
doubling the sea ice contribution at the transformation maximum from 0.86 to 1.84 Sv.""",
"""comparison is now like-for-like. The result is shown in the new Fig. 8 and discussed in the
Discussion. Restricting the integration to the ice-covered sector raises the haline share, as
expected, moving the thermal fraction of the dense-class transformation from 69% to 61% and roughly
doubling the sea ice contribution at the transformation maximum from 0.86 to 1.84 Sv.

![](letter_figs/figR1_wmt_domain_sensitivity.png)

**Figure R1 (manuscript Fig. 8).** Transformation integrated south of 60°S, the domain used in the
submitted version (top row), and over the seasonal sea ice zone alone, which reproduces the domain
of Pellichero et al. to within 2.5% in area (bottom row). The thermal share at the transformation
maximum and the area of each domain are given in each panel. The haline share increases in the
restricted domain, as expected, but the contrast between thermally dominated interglacials and
haline dominated glacials is unchanged.""",
  'r1-figR1')

# ------------------------------------------------------------------ R2 major 1 figure
E("""contribution at the transformation maximum roughly doubles from 0.86 to 1.84 Sv. This is shown in
the new Fig. 8.""",
"""contribution at the transformation maximum roughly doubles from 0.86 to 1.84 Sv. This is shown in
Fig. R1 above, reproduced as Fig. 8 of the manuscript.""",
  'r2-figR1-ref')

# ------------------------------------------------------------------ R2: SAM section, figures moved
E("""We have shortened the section, removing the repetition that Reviewer 1 also flagged and cutting the
speculative proxy passage. We have, however, kept the figures in the main text, and we hope the
reviewer will accept our reasoning. Reviewer 3 independently judged this to be the most
generalisable part of the paper, describing it as showing that wind changes affect "both the push
and the pull driving the overturning". Given that the two reviewers take opposite views, we have
tried to satisfy both by tightening the presentation while keeping the material accessible. If the
editor prefers, we are willing to move Figs 5 and 6 to the supplement.""",
"""We have done this. All of the Southern Annular Mode figures, including Figs 5 and 6, have been
moved to the Supplementary Information, so the main text no longer carries any figure devoted to
the mode. We have also shortened the section, removing the repetition that Reviewer 1 flagged and
cutting the speculative passage about marine proxy reconstruction. What remains in the main text is
a compact account of the result, with the supporting composites available to readers who want them.

We note that Reviewer 3 took the opposite view of this material, describing it as the more
generalisable part of the study. Moving the figures to the supplement while keeping a concise
statement of the result in the main text seemed to us the arrangement most likely to satisfy both
readings, and we are content with the outcome.""",
  'r2-sam-moved')

# ------------------------------------------------------------------ R2: units comparison
E("""This is a fair criticism. We have added a statement of the comparison in equivalent surface density
tendency terms to the Results: expressed that way, the freshwater contribution is the larger of the
two wherever sea ice is present, and the heat contribution is larger only in the ice-free sector,
which is the same partition that governs the mean state. We have kept the native units in the
figures themselves, because the heat and freshwater fields are what is diagnosed and readers may
wish to compare them against other studies in conventional units, but the text now makes the
relative magnitudes explicit. If the reviewer would prefer the figures themselves recast in
buoyancy units, we will do so.""",
"""This is a fair criticism. We have added a statement of the comparison in equivalent surface density
tendency terms to the Results: expressed that way, the freshwater contribution is the larger of the
two wherever sea ice is present, and the heat contribution is larger only in the ice-free sector,
which is the same partition that governs the mean state. We have kept the native units in the
figures themselves, which are now supplementary, because the heat and freshwater fields are what
the model diagnoses and readers may wish to compare them against other studies in conventional
units. The text now makes the relative magnitudes explicit. If the reviewer would prefer the
figures recast in buoyancy units, we will do so.""",
  'r2-units')

# ------------------------------------------------------------------ R3: SAM strengthened
E("""> *I find the SAM wind results interesting and I think more generalisable. ... This suggests changes
> in the wind patterns effect both the push and pull driving the overturning.*

We are grateful for this observation and have kept the section in the main text partly on the
strength of it, noting that Reviewer 2 suggested the opposite. We have adopted the reviewer's
framing of the wind acting on both the divergence and the salt pump.""",
"""> *I find the SAM wind results interesting and I think more generalisable. The results suggest
> changes in wind shift/strength impacts not just the ekman divergence i.e. upwelling of deepwaters,
> but also the salt pump via sea ice export, and thus the formation of deepwaters. This suggests
> changes in the wind patterns effect both the push and pull driving the overturning.*

We are grateful for this observation, which identified a mechanism our original analysis had not
isolated, and we have carried out new work to test it directly. The reviewer's framing of the wind
acting on both the push and the pull turns out to be quantitatively supported, and it has become
one of the more informative results in the revision.

We separated the two routes explicitly. For the same high and low SAM composites, we computed the
zonally integrated northward Ekman transport at 60°S from the simulated zonal wind stress, which
measures the push, and the coastal brine input integrated south of 65°S, which measures the pull.
Both were computed for all five climate states.

![](letter_figs/figR4_sam_push_pull.png)

**Figure R4 (new Supplementary Figure).** (a) The positive phase strengthens the Ekman transport at
60°S by 8.0--12.8 Sv in every climate state. (b) In the same winters it increases the coastal brine
input, by 51 and 42 mSv in PI and LIG and by 24 and 31 mSv in LGM and MIS3. (c) The two responses
scale together across the five states (r = 0.77). (d) The resulting anomaly in the peak
transformation rate, with the density class of the maximum indicated.

Three results follow, and all three are new relative to the submitted version.

The push and the pull are not independent expressions of the same forcing but two halves of one
mechanism. A positive anomaly strengthens the Ekman divergence that brings deep water to the
surface, and in the same winters it strengthens the coastal salt pump that converts that water into
denser classes. Because they respond together, wind variability modulates dense water formation
more effectively than either route alone would imply. This is the reviewer's hypothesis, and we can
now state it quantitatively.

The balance between the routes is state dependent, and in a way we did not anticipate. The Ekman
response is largest in the interglacials (11.2 Sv in PI, 12.8 Sv in LIG) and weakest at the Last
Glacial Maximum (8.0 Sv), even though the glacial mean wind stress is stronger. Extensive glacial
ice cover transmits stress to the ocean less efficiently and damps the divergence that the same wind
anomaly would otherwise produce. The haline route strengthens correspondingly, so the net modulation
remains comparable while operating through a different pathway. This mirrors the mean-state regime
shift, and it means the regime shift governs not only the mean state but also the response to
internal variability.

The spatial patterns show the mechanism directly.

![](letter_figs/figR5_sam_wind_ice.png)

**Figure R5 (new Supplementary Figure).** Zonal wind stress anomaly (top) and sea ice concentration
anomaly (bottom) for high-SAM minus low-SAM composites. The westerlies strengthen over the
circumpolar belt in every state. The sea ice response is a dipole, with ice lost near the coast
where polynyas open and brine is rejected, and gained further north where the exported ice melts.
The dipole sits further north and is substantially stronger in the glacial states.

These results are now described in the main text, and the supporting figures are in the
Supplementary Information.""",
  'r3-sam-strengthened')

# ------------------------------------------------------------------ R3: MOC + age figures
E("""The overturning streamfunction, diagnosed online by the model, is now shown in the new Fig. 6. The
abyssal cell weakens from −10.0 Sv in PI, −10.1 Sv in MH and −9.2 Sv in LIG to −2.4 Sv in LGM and
−1.6 Sv in MIS3, a reduction of roughly 75–85%, while the upper cell changes comparatively little
(19.1 Sv in PI against 20.0 and 21.7 Sv in the glacials). The glacial reorganisation in these
simulations is therefore concentrated in the abyssal cell.""",
"""The overturning streamfunction, diagnosed online by the model, is now shown in the new Fig. 6. The
abyssal cell weakens from −10.0 Sv in PI, −10.1 Sv in MH and −9.2 Sv in LIG to −2.4 Sv in LGM and
−1.6 Sv in MIS3, a reduction of roughly 75–85%, while the upper cell changes comparatively little
(19.1 Sv in PI against 20.0 and 21.7 Sv in the glacials). The glacial reorganisation in these
simulations is therefore concentrated in the abyssal cell.

![](letter_figs/figR2_moc_5exps.png)

**Figure R2 (manuscript Fig. 6).** Global overturning streamfunction for the five climate states
(top) and anomalies relative to PI (bottom). The blue abyssal cell, representing northward
spreading of southern-sourced bottom water, nearly disappears in the glacial states while the upper
cell is largely maintained.""",
  'r3-figR2')

E("""Global ideal age is now shown in the new Fig. 7, as basin-mean vertical profiles and zonal-mean
sections. Below 2000 m the ageing is strongly basin dependent: the Atlantic increases only modestly
from 377 to 439 years (LGM) and 508 years (MIS3), while the Southern Ocean more than doubles from
381 to 789 and 844 years and the Pacific goes from 847 to 1291 and 1314 years. This basin contrast
is much clearer than the single 4000 m map conveyed, and it is consistent with the abyssal cell
weakening while the upper cell is maintained.""",
"""Global ideal age is now shown in the new Fig. 7, as basin-mean vertical profiles and zonal-mean
sections. Below 2000 m the ageing is strongly basin dependent: the Atlantic increases only modestly
from 377 to 439 years (LGM) and 508 years (MIS3), while the Southern Ocean more than doubles from
381 to 789 and 844 years and the Pacific goes from 847 to 1291 and 1314 years. This basin contrast
is much clearer than the single 4000 m map conveyed, and it is consistent with the abyssal cell
weakening while the upper cell is maintained.

![](letter_figs/figR3_age_global.png)

**Figure R3 (manuscript Fig. 7).** Basin-mean ideal age profiles (a-d), mean age below 2000 m (e),
and zonal-mean sections for the five states (f-j). The glacial ageing is concentrated below about
2000 m and is largest in the deep Pacific and Southern Ocean, while the Atlantic, which remains
ventilated from the north, changes comparatively little.""",
  'r3-figR3')

missing = []
for old, new, label in edits:
    n = s.count(old)
    if n != 1:
        missing.append((label, n))
        continue
    s = s.replace(old, new, 1)

if missing:
    print('FAILED:')
    for label, n in missing:
        print(f'  {label}: found {n}')
    sys.exit(1)

P.write_text(s)
print(f'applied {len(edits)} letter edits')
