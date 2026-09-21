#!/usr/bin/env python
"""
Restructure the response letter's figures:
  - per-reviewer numbering: R1.1, R1.2 ... for Reviewer 1; R2.1 ... for Reviewer 2, etc.
  - the same underlying figure may appear under several reviewers, each with its
    own number, because several reviewers asked the same question
  - add figures to comments that currently carry only text, wherever a figure helps
"""
import sys
from pathlib import Path

P = Path('RESPONSE_LETTER.md')
s = P.read_text()
edits = []


def E(old, new, label):
    edits.append((old, new, label))


IMG = 'letter_figs'

# =====================================================================
# REVIEWER 1
# =====================================================================

# R1 G1 -- already has the domain figure; renumber and add the convection figure
E(f"""![]({IMG}/figR1_wmt_domain_sensitivity.png)

**Figure R1 (manuscript Fig. 8).** Transformation integrated south of 60°S, the domain used in the
submitted version (top row), and over the seasonal sea ice zone alone, which reproduces the domain
of Pellichero et al. to within 2.5% in area (bottom row). The thermal share at the transformation
maximum and the area of each domain are given in each panel. The haline share increases in the
restricted domain, as expected, but the contrast between thermally dominated interglacials and
haline dominated glacials is unchanged.""",
  f"""![]({IMG}/figR1_wmt_domain_sensitivity.png)

**Figure R1.1 (manuscript Fig. 8).** Transformation integrated south of 60°S, the domain used in
the submitted version (top row), and over the seasonal sea ice zone alone, which reproduces the
domain of Pellichero et al. to within 2.5% in area (bottom row). The thermal share at the
transformation maximum and the area of each domain are given in each panel. The haline share
increases in the restricted domain, as expected, but the contrast between thermally dominated
interglacials and haline dominated glacials is unchanged.

We also diagnosed directly where the model forms its dense water, which bears on how far the
pre-industrial state can be trusted.

![]({IMG}/figR6_convection_sites.png)

**Figure R1.2 (new analysis).** (a-e) Winter mixed layer depth for the five states, with the 400 m
contour in orange. (f) In the pre-industrial state the Weddell sector accounts for 70% of the area
with a mixed layer deeper than 400 m and 97% of the area deeper than 600 m, and the convection sits
in the interior of the gyre near 68°S rather than over the shelf. (g) The deep-convection area
contracts to roughly one fifth of its pre-industrial value in the glacial states. (h) The deepest
winter mixed layer shoals from 991 m in PI to 819 m and 645 m in the glacials.""",
  'r1-g1-figs')

# R1#10 -- MLD shallower in LGM: add the figure
E("""1.6×10¹¹ m² in MIS3, the maximum mixed-layer depth decreases from 991 m to 819 m and 645 m, and
mixed layers deeper than 600 m essentially disappear in both glacial states. The text now says
explicitly that the residual glacial signals are genuinely shallower than their interglacial
counterparts.""",
  f"""1.6×10¹¹ m² in MIS3, the maximum mixed-layer depth decreases from 991 m to 819 m and 645 m, and
mixed layers deeper than 600 m essentially disappear in both glacial states. The text now says
explicitly that the residual glacial signals are genuinely shallower than their interglacial
counterparts. Panels (g) and (h) of Figure R1.2 above make the point quantitatively: both the area
of deep convection and the depth of the deepest mixed layer fall in the glacial states, so the
reviewer's reading of the shading is correct and is now reflected in the text.""",
  'r1-10-fig')

# R1#15 -- common axis figure
E("""— Yes. Supplementary Fig. S5 now shows the same transformation curves on common vertical and
horizontal axes, so magnitudes and density positions can be compared directly across climate states
and regions. We have kept the independently scaled version in the main text because the shape of
the individual curves is otherwise hard to read, and the main text now points to the common-axis
version explicitly.""",
  f"""— Yes. Supplementary Fig. S5 now shows the same transformation curves on common vertical and
horizontal axes, so magnitudes and density positions can be compared directly across climate states
and regions. We have kept the independently scaled version in the main text because the shape of
the individual curves is otherwise hard to read, and the main text now points to the common-axis
version explicitly.

![]({IMG}/figS_common_axis-1.png)

**Figure R1.3 (new Supplementary Fig. S5).** Winter transformation for the four sectors and five
climate states on a common vertical and horizontal axis. Plotted this way the glacial weakening of
the transformation maximum and its shift to denser classes are directly comparable between panels,
which the independently scaled main-text version does not allow.""",
  'r1-15-fig')

# R1#23 -- where does deep water form
E("""— We have added
this diagnosis to the Results: in the pre-industrial state the Weddell sector accounts for 70% of
the area with winter mixed layers deeper than 400 m south of 55°S and 97% of the area deeper than
600 m, centred near 68°S in the gyre interior. The formation region is therefore the open Weddell
gyre rather than the shelf, which is directly relevant to the model limitation discussed under G1.""",
  """— We have added
this diagnosis to the Results, and it is shown in Figure R1.2 above. In the pre-industrial state the
Weddell sector accounts for 70% of the area with winter mixed layers deeper than 400 m south of
55°S and 97% of the area deeper than 600 m, centred near 68°S in the gyre interior. The formation
region is therefore the open Weddell gyre rather than the shelf, which is directly relevant to the
model limitation discussed under G1.""",
  'r1-23-fig')

# R1#26 -- observational comparison figure reference
E("""— We have added the
quantitative comparison to the text and to the new Fig. 8, which shows our transformation
recomputed over the Pellichero domain alongside the standard domain.""",
  """— We have added the
quantitative comparison to the text and to the new Fig. 8, reproduced as Figure R1.1 above, which
shows our transformation recomputed over the Pellichero domain alongside the standard domain.""",
  'r1-26-fig')

# =====================================================================
# REVIEWER 2 -- currently has no figures at all
# =====================================================================

E("""contribution at the transformation maximum roughly doubles from 0.86 to 1.84 Sv. This is shown in
Fig. R1 above, reproduced as Fig. 8 of the manuscript.""",
  f"""contribution at the transformation maximum roughly doubles from 0.86 to 1.84 Sv.

![]({IMG}/figR1_wmt_domain_sensitivity.png)

**Figure R2.1 (manuscript Fig. 8).** Transformation integrated south of 60°S, the domain used in
the submitted version (top row), and over the seasonal sea ice zone alone, which reproduces the
domain of Pellichero et al. to within 2.5% in area (bottom row). Restricting the integration to the
ice-covered sector raises the haline share, as the reviewer would expect, but the contrast between
thermally dominated interglacials and haline dominated glacials survives intact.""",
  'r2-fig1')

E("""What differs is where dense water is made. In the model, pre-industrial dense water forms by
open-ocean convection in the interior of the Weddell gyre near 68°S rather than by the sequence of
shelf water formation and downslope overflow that operates in the real ocean. Open-ocean convection
exposes a large area to the atmosphere and so recruits an excessive thermal contribution while
producing water that is not dense enough, which is precisely the offset in density class described
above.""",
  f"""What differs is where dense water is made. In the model, pre-industrial dense water forms by
open-ocean convection in the interior of the Weddell gyre near 68°S rather than by the sequence of
shelf water formation and downslope overflow that operates in the real ocean. Open-ocean convection
exposes a large area to the atmosphere and so recruits an excessive thermal contribution while
producing water that is not dense enough, which is precisely the offset in density class described
above.

![]({IMG}/figR6_convection_sites.png)

**Figure R2.2 (new analysis).** Where the model actually convects. (a-e) Winter mixed layer depth,
400 m contour in orange. (f) In the pre-industrial state 70% of the area with a mixed layer deeper
than 400 m, and 97% of the area deeper than 600 m, lies in the Weddell sector, centred near 68°S in
the open gyre rather than over the shelf. (g, h) The deep-convection area and the maximum mixed
layer depth both contract sharply in the glacial states.""",
  'r2-fig2')

E("""Second, the mechanism is the ice cover itself rather than the convection style: insulation of the
surface and concentration of brine rejection follow from the areal expansion of sea ice, and both
would operate in a model that formed its dense water on the shelf.""",
  f"""Second, the mechanism is the ice cover itself rather than the convection style: insulation of the
surface and concentration of brine rejection follow from the areal expansion of sea ice, and both
would operate in a model that formed its dense water on the shelf.

The consequences for the large-scale circulation are visible in the overturning itself.

![]({IMG}/figR2_moc_5exps.png)

**Figure R2.3 (manuscript Fig. 6).** Global overturning streamfunction (top) and anomalies relative
to PI (bottom). The abyssal cell that carries southern-sourced bottom water weakens from about
10 Sv in the interglacials to 2.4 and 1.6 Sv in LGM and MIS3, while the upper cell is largely
unchanged. The glacial reorganisation is concentrated in the cell our surface analysis addresses.""",
  'r2-fig3')

# =====================================================================
# REVIEWER 3 -- renumber existing figures
# =====================================================================

E("""![](letter_figs/figR4_sam_push_pull.png)

**Figure R4 (new Supplementary Figure).**""",
  f"""![]({IMG}/figR4_sam_push_pull.png)

**Figure R3.1 (new Supplementary Figure).**""",
  'r3-fig1')

E("""![](letter_figs/figR5_sam_wind_ice.png)

**Figure R5 (new Supplementary Figure).**""",
  f"""![]({IMG}/figR5_sam_wind_ice.png)

**Figure R3.2 (new Supplementary Figure).**""",
  'r3-fig2')

E("""![](letter_figs/figR2_moc_5exps.png)

**Figure R2 (manuscript Fig. 6).** Global overturning streamfunction for the five climate states
(top) and anomalies relative to PI (bottom). The blue abyssal cell, representing northward
spreading of southern-sourced bottom water, nearly disappears in the glacial states while the upper
cell is largely maintained.""",
  f"""![]({IMG}/figR2_moc_5exps.png)

**Figure R3.3 (manuscript Fig. 6).** Global overturning streamfunction for the five climate states
(top) and anomalies relative to PI (bottom). The blue abyssal cell, representing northward
spreading of southern-sourced bottom water, nearly disappears in the glacial states while the upper
cell is largely maintained.""",
  'r3-fig3')

E("""![](letter_figs/figR3_age_global.png)

**Figure R3 (manuscript Fig. 7).** Basin-mean ideal age profiles (a-d), mean age below 2000 m (e),
and zonal-mean sections for the five states (f-j). The glacial ageing is concentrated below about
2000 m and is largest in the deep Pacific and Southern Ocean, while the Atlantic, which remains
ventilated from the north, changes comparatively little.""",
  f"""![]({IMG}/figR3_age_global.png)

**Figure R3.4 (manuscript Fig. 7).** Basin-mean ideal age profiles (a-d), mean age below 2000 m
(e), and zonal-mean sections for the five states (f-j). The glacial ageing is concentrated below
about 2000 m and is largest in the deep Pacific and Southern Ocean, while the Atlantic, which
remains ventilated from the north, changes comparatively little.""",
  'r3-fig4')

# R3 main critique -- add the convection figure, since this is his central point
E("""We confirmed the reviewer's suspicion diagnostically. In the pre-industrial simulation the Weddell
sector accounts for 70% of the area with a winter mixed layer deeper than 400 m south of 55°S and
97% of the area deeper than 600 m, centred near 68°S in the interior of the gyre rather than over
the continental shelf.""",
  f"""We confirmed the reviewer's suspicion diagnostically. In the pre-industrial simulation the Weddell
sector accounts for 70% of the area with a winter mixed layer deeper than 400 m south of 55°S and
97% of the area deeper than 600 m, centred near 68°S in the interior of the gyre rather than over
the continental shelf.

![]({IMG}/figR6_convection_sites.png)

**Figure R3.5 (new analysis).** (a-e) Winter mixed layer depth with the 400 m contour in orange.
(f) The pre-industrial sector breakdown, showing that convection is concentrated in the Weddell
gyre interior rather than on the shelf, which is the bias the reviewer identified. (g, h) The
glacial contraction of both the deep-convection area and the maximum mixed-layer depth.""",
  'r3-fig5')

# R3 domain test -- add the domain figure
E("""We took the first of these suggestions and report the results in full under Reviewer 2's major
issue 1 and in the new Fig. 8. In summary: 42.7% of our published domain lies outside the September
sea ice edge; restricting to their domain moves the thermal share of dense-class transformation from
69% to 61%; and, more importantly, our thermally dominated maximum sits at γ_n ≈ 27.3 kg m⁻³ while
their haline-dominated cell occupies γ_n = 27.9–28.8 kg m⁻³, so the two analyses largely concern
different water masses. In the densest classes we do populate, our result is haline dominated and
therefore agrees with theirs.""",
  f"""We took the first of these suggestions. In summary: 42.7% of our published domain lies outside the
September sea ice edge; restricting to their domain moves the thermal share of dense-class
transformation from 69% to 61%; and, more importantly, our thermally dominated maximum sits at
γ_n ≈ 27.3 kg m⁻³ while their haline-dominated cell occupies γ_n = 27.9–28.8 kg m⁻³, so the two
analyses largely concern different water masses. In the densest classes we do populate, our result
is haline dominated and therefore agrees with theirs.

![]({IMG}/figR1_wmt_domain_sensitivity.png)

**Figure R3.6 (manuscript Fig. 8).** The like-for-like comparison the reviewer asked for.
Transformation integrated over the published domain south of 60°S (top) and over the seasonal sea
ice zone that reproduces the observational domain (bottom). The haline share rises in the
restricted domain, but the glacial-interglacial contrast is unaffected, which is why we conclude
the regime shift is not an artefact of the biased pre-industrial end member.""",
  'r3-fig6')

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
print(f'applied {len(edits)} figure edits')
