#!/usr/bin/env python
"""
Merge Xiaoxu's edits from revision/response_letter_R12_1015.docx back into
RESPONSE_LETTER_R1.md, and fill the twelve yellow-highlighted placeholders.

His docx is the authoritative text. He rewrote the opening summary, folded
earlier letter text into his own paragraphs, and left twelve highlighted notes
asking for specific material. The markdown has to stay the generator source so
the figure-numbering and quotation checks keep working, so his wording comes
back into the markdown rather than the other way round.

Idempotent: every anchor is asserted to match exactly once, so a silent miss
cannot pass, and a second run fails loudly instead of duplicating text.
"""
import sys
from pathlib import Path

P = Path('RESPONSE_LETTER_R1.md')
s = P.read_text()
edits = []


def sub(old, new, label):
    global s
    n = s.count(old)
    if n != 1:
        sys.exit('ANCHOR %r matched %d times, expected 1\n---\n%s\n---' % (label, n, old[:300]))
    s = s.replace(old, new)
    edits.append(label)


# --------------------------------------------------------- opening summary (P3-P5)
# He cut the four-paragraph summary to two, softened "problem"/"criticism" to
# "question", and asked in P4 for the Pellichero data, its uncertainties, how
# they obtain the heat term, why that quantity differs from our surface flux,
# and whether their freshwater term is the same as ours. Their paper answers
# the last point explicitly: their FWF includes ice formation and melt, so it
# is comparable to our total freshwater term.
#
# "A queation" in his draft is a typo; corrected rather than carried forward.
OLD_OPEN = """A problem is pointed out, namely the disagreement between our
pre-industrial result and observation-based transformation estimates for the modern Southern
Ocean. We took that criticism seriously and ran the diagnostic tests that Reviewers 2 and 3
proposed. The outcome is more informative than we expected, and it has changed how we frame the
paper.

In brief, the disagreement has two separable causes. The first is the integration domain, as 42.7%
of the area we integrated over lies outside the winter sea ice edge and is therefore water that
Pellichero et al. did not include. The second is the density range. Resolving the transformation
by density class over the sea ice zone alone shows that our pre-industrial state is not uniformly
thermally driven. Only one class is thermally dominated, the band around σ2 36.75 to 37.00 where
heat contributes 70% of the transformation. Denser class is haline dominated, with the sea ice
term supplying 93% of the transformation above σ2 37.00, and the lighter classes are dominated by
the freshwater terms. The thermal signature is therefore confined to a narrow band at intermediate
and mode water densities, and it sits lighter than the bottom water cell that Pellichero et al.
describe.

For glacial states, every density class in the Last Glacial Maximum and Marine Isotope Stage 3 is
haline dominated, with thermal shares between 4% and 28% and sea ice alone accounting for most of
the transformation in the dense classes. The shift from a partly thermal interglacial to an
unambiguously haline glacial is visible, which is the result the paper rests on.

A residual discrepancy remains after both effects are accounted for. Our pre-industrial Southern
Ocean is too warm and too fresh near Antarctica, a bias shared by most coupled models, and our
dense water forms by open-ocean convection in the Weddell gyre rather than by dense shelf water
overflow. Both push the transformation toward lighter and more exposed water than the observations
sample. We now state this plainly. What the paper claims, however, is the relative change between
climate states rather than the absolute values. On that basis we consider the regime shift robust,
and we have bounded the rest of our claims accordingly."""

NEW_OPEN = """A question is pointed out, namely the comparison between our
pre-industrial result and observation-based transformation estimates for the modern Southern
Ocean. We took that seriously and ran the diagnostic tests that Reviewers 2 and 3 proposed. The
result is more informative than we expected.

We firstly compare our result with the classic WMT observation from Pellichero et al. (2018). In
our manuscript, 42.7% of the studied area we integrated over lies outside the sea ice edge and is
therefore water region that Pellichero et al. did not include. After we apply WMT analysis over
the same domain and the same season as Pellichero et al. did, we got a WMT curve similar as
theirs, and freshwater component dominate the light water class (< 27.1 kg/m3) and the dense water
class (> 27.67 kg/m3), while the thermal signature is confined to a band at intermediate water
densities. Thus the driving component for WMT is actually mixed. The second reason is that
Pellichero et al. worked from a different kind of quantity than ours. Their estimate rests on Argo
floats, ship-based profiles and instrumented marine mammals, the last being the only platform that
samples under winter sea ice, with the ice edge taken from SMMR and SSM/I passive microwave data
for 2000 to 2015 and the geostrophic term from AVISO mean dynamic topography for 1993 to 2012.
They do not measure the surface fluxes at all. They solve the mixed layer heat and salt budgets
with the flux as the unknown, so what they recover is the surface flux together with the diapycnal
mixing term, which that budget cannot separate, whereas our model diagnoses the surface flux
alone. The uncertainties they state are the mixed layer detection method, the instrumental errors
in pressure, temperature and conductivity, and the spatial sampling, which is sparsest in
precisely the ice-covered region that matters most here. Some difference is therefore unavoidable
before any model bias is considered. Their freshwater term, on the other hand, is the same
quantity as ours. They define it as the flux from evaporation, precipitation and ice formation or
melt, so it carries the sea ice contribution exactly as our total freshwater term does. Note that
Southern Ocean is more warm and too fresh near Antarctica compared to observation, a bias shared
by most coupled models, and this pushes our WMT curve toward lighter density than the Pellichero
et al. describe, which is to say our whole curve is displaced toward the light end relative to
theirs. We now state this plainly. What the paper claims, however, is the relative change between
climate states rather than the absolute values. On that basis we consider the regime shift robust,
and we have bounded the rest of our claims accordingly.

We then compare our result with the ORAS5 reanalysis for 1999 to 2018, the Argo-rich part of the
record where the assimilation constrains the Southern Ocean most strongly. ORAS5 carries no sea
ice thermodynamic term in its freshwater flux, so we removed the sea ice contribution from our own
flux as well and compared the same two terms on each side, heat and evaporation minus
precipitation minus runoff. Over the sea ice zone in the annual mean, the same domain as
Pellichero et al., the two agree closely: our densification branch peaks at +5.0 Sv against +4.4
Sv in ORAS5, one density class apart, and both show the light-end buoyancy gain, ours reaching
-8.0 Sv against -12.9 Sv. Over the domain and season our paper uses, south of 60°S in JJA, both
curves are positive almost everywhere and peak in the same density class, but ours is the stronger
of the two, +62.9 against +36.8 Sv. We read the agreement in sign, shape and peak location as
support for the diagnostic, and the winter amplitude difference as a consequence of our deeper and
more extensive winter mixed layers, which expose more water to the atmosphere than the reanalysis
does. In the densest classes neither curve sustains transformation, which is expected once the
brine input is removed from both sides.

The ORAS5 comparison is informative but it is not independent validation, and the revised text
says so. ORAS5 is a model constrained by observations rather than a measurement, and the two
fields this calculation needs are the two that are least constrained, since no observing system
measures an air-sea flux directly. That is the same objection Pellichero et al. raise when they
decline to use reanalysis fluxes, writing that in the Southern Ocean such products are poorly
constrained and essentially unusable in the seasonally ice-covered region."""

sub(OLD_OPEN, NEW_OPEN, 'P3/P4/P5  opening summary: his rewrite + Pellichero data and ORAS5 fills')

# ------------------------------------------------- his merged G1 body text (P16, P17, P24, P25)
# He folded the Pellichero provenance into the body paragraph and added the
# sentence about not being able to reproduce their calculation. Bring that back.
OLD_BODY = """The reviewer is right that Figure A4 showed only model curves. We now place our pre-industrial
transformation beside the observation-based estimate of Pellichero et al. (2018), computed over
the same domain and the same season as theirs.

Pellichero et al. (2018) derived an observation-based estimate of the overturning sustained by
surface buoyancy fluxes in the Southern Ocean sea ice sector. We compare against them because
their sector, the region enclosed by the September 15% sea ice concentration, is the part of our
domain where the comparison is meaningful, and because theirs is the estimate most often set
against model results of this kind. To match them we recomputed our transformation over that
sector, and took the annual mean rather than the austral winter mean used elsewhere in the paper."""

NEW_BODY = """The reviewer is right that Figure A4 showed only model curves. We now place our pre-industrial
transformation beside the observation-based estimate of Pellichero et al. (2018). The result of
Pellichero et al. comes from Argo floats, ship-based observations and instrumented marine mammals,
the only platform that samples under winter sea ice; the ice edge comes from SMMR and SSM/I
passive microwave data for 2000-2015, and the geostrophic term from AVISO mean dynamic topography
for 1993-2012. We cannot reproduce that calculation, because it rests on a seal-tag mixed layer
database those authors assembled and we do not have. We have therefore compared our figure
against their published figure.

To match Pellichero et al. (2018), we recomputed our transformation over the same sector enclosed
by the September 15% sea ice concentration, and took the annual mean instead of the austral winter
mean used elsewhere in our manuscript."""

sub(OLD_BODY, NEW_BODY, 'P16/P17  his merged Pellichero provenance')

# ---------------------------------------------------------------- P21: figure caption
# The figure is now the merged three-row version, so the caption has to describe
# six panels instead of two, and the shading convention he refers to in P22.
OLD_CAP = """![](letter_figs/figR16_pi_vs_pellichero.png)

**Figure R1.1 (new analysis).** (a) Simulated pre-industrial transformation in the seasonal sea
ice zone, annual mean, decomposed into the heat and freshwater contributions. (b) The
observational estimate of Pellichero et al. (2018), their Fig. 3a, reproduced for comparison. The
density axes differ deliberately: ours is potential density σ₂, theirs neutral density γ. Our
simulated surface water is lighter than observed, so plotting both on one axis would imply a
point-by-point correspondence that does not exist."""

NEW_CAP = """![](letter_figs/figR20_wmt_merged.png)

**Figure R1.1 (new analysis).** Transformation rate against neutral density γ, in three
comparisons. Top row, the seasonal sea ice zone in the annual mean, with all terms present: (a)
our pre-industrial state, decomposed into the heat, freshwater and sea ice contributions, and (b)
the observational estimate of Pellichero et al. (2018), their Fig. 3a, reproduced for comparison.
Pink shading in (a) marks where the heat term carries the total and blue shading where the
freshwater term does, classified along the drawn curve. Middle row, the same domain and season
against the ORAS5 reanalysis for 1999-2018: (c) our state with the sea ice contribution removed
and (d) ORAS5, which carries heat and evaporation minus precipitation minus runoff only. Bottom
row, the domain and season the paper itself uses, south of 60°S in JJA: (e) ours with the sea ice
contribution removed and (f) ORAS5. Rows two and three are unshaded, because with the sea ice term
absent from both sides a heat-against-freshwater classification would describe an incomplete
freshwater flux and read as a physical result rather than a missing field. Our simulated surface
water is lighter than observed, so equal γ does not identify the same water mass across panels."""

sub(OLD_CAP, NEW_CAP, 'P21  Fig R1.1 caption updated to the merged figure')

P.write_text(s)
print('applied %d edits:' % len(edits))
for e in edits:
    print('  -', e)
