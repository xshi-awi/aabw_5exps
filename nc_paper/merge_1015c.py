#!/usr/bin/env python
"""
Third merge batch: Reviewer 2, Major issue 1 (docx paragraphs 222-229).

His highlighted instruction there is that the five paragraphs of the old R2
answer are stale and should be replaced by a rewrite of what we now say to
Reviewer 1, with Fig R1.1 copied in as Fig R2.1.

The rewrite follows the R1 order, because that is the order he settled on:
the observational comparison first, then the reanalysis check, then what is
left over as genuine bias, then whether the main result survives. The
substantive numbers are unchanged; what changes is that the answer now leads
with the two comparisons rather than with the domain argument alone.
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


OLD = """The reviewer is right that the original single sentence was inadequate, and right to insist that
the answer matters for the paper's main claim. We therefore tested it directly rather than
speculating. Three findings emerged, and we report all three, including the one that is
unfavourable to us.

**The domains are less similar than they appear.** Pellichero et al. define their sector as the
region enclosed by the September 15% sea ice contour, so its northern boundary is an ice contour
that varies with longitude, not a latitude circle. Our published integral covers everything south
of 60°S. In our pre-industrial simulation, 42.7% of that area lies outside the September ice
contour and is therefore water their analysis does not include. It is also the part of the domain
that stays exposed to the atmosphere year round, and taken alone it is 89–91% thermally driven in
the dense classes. Recomputing our transformation over a sea ice sector defined exactly as theirs, which reproduces their domain to within 2.5% in area, sharpens the picture considerably. Resolved by density class, only one class in our pre-industrial state is thermally dominated, σ₂ = 36.75–37.00 kg m⁻³ at 70%, and every other class is haline dominated, falling to 6–7% thermal in the densest classes. Both glacial states are haline dominated in every class, between 4% and 28%. The regime shift the paper reports is therefore not only preserved inside the observational domain, it is clearer there than in the wider domain we originally used.

![](letter_figs/figR1_wmt_domain_sensitivity.png)"""

NEW = """The reviewer is right that the original single sentence was inadequate, and right to insist that
the answer matters for the paper's main claim. We therefore tested it directly rather than
speculating, in two ways: we recomputed our transformation over the domain and season Pellichero
et al. actually used and placed the two curves side by side, and we computed an independent
transformation from the ORAS5 reanalysis over the same domains. Several findings emerged, and we
report all of them, including the ones that are unfavourable to us.

![](letter_figs/figR20_wmt_merged.png)

**Figure R2.1 (new analysis).** Transformation rate against neutral density γ, in three
comparisons. Top row, the seasonal sea ice zone in the annual mean, with all terms present: (a)
our pre-industrial state, decomposed into the heat, freshwater and sea ice contributions, and (b)
the estimate of Pellichero et al. (2018), their Fig. 3a, reproduced for comparison. Pink shading
in (a) marks where the heat term carries the total and blue shading where the freshwater term
does. Middle row, the same domain and season against ORAS5 for 1999-2018: (c) ours with the sea
ice contribution removed and (d) ORAS5, which carries heat and evaporation minus precipitation
minus runoff only. Bottom row, the domain and season the paper uses, south of 60°S in JJA: (e)
ours with the sea ice contribution removed and (f) ORAS5. Rows two and three are unshaded, because
with the sea ice term absent from both sides a heat-against-freshwater classification would
describe an incomplete freshwater flux.

**Compared like for like, the curves agree in structure and the forcing is mixed rather than
thermal.** Recomputed over their sector and their season, our transformation shares the structure
of theirs: buoyancy gain at the lighter classes, a zero crossing, and densification at the heavier
classes. Ours reaches a minimum of -9.2 Sv, crosses zero, and peaks at +6.6 Sv, against a minimum
near -22 Sv and a densification peak of 5±5 Sv in their estimate, so the sign structure and the
order of magnitude of the densification branch agree. Resolved by density, the freshwater term
dominates the light classes below γ 27.1 and the dense classes above γ 27.67, while the thermal
signature is confined to a band at intermediate densities. The characterisation of our
pre-industrial state as thermally dominated, which the reviewer reasonably took from our text, is
therefore not what the comparable calculation shows. The driving component is mixed, and the
thermal part of it sits in a narrow band of intermediate and mode water densities rather than
across the whole range.

**The domains are less similar than they appear.** Pellichero et al. define their sector as the
region enclosed by the September 15% sea ice contour, so its northern boundary is an ice contour
that varies with longitude, not a latitude circle. Our published integral covers everything south
of 60°S. In our pre-industrial simulation, 42.7% of that area lies outside the September ice
contour and is therefore water their analysis does not include. It is also the part of the domain
that stays exposed to the atmosphere year round, and taken alone it is 89–91% thermally driven in
the dense classes. Recomputing our transformation over a sea ice sector defined exactly as theirs,
which reproduces their domain to within 2.5% in area, sharpens the picture considerably. Resolved
by density class, only one class in our pre-industrial state is thermally dominated, σ₂ =
36.75–37.00 kg m⁻³ at 70%, and every other class is haline dominated, falling to 6–7% thermal in
the densest classes. Both glacial states are haline dominated in every class, between 4% and 28%.
The regime shift the paper reports is therefore not only preserved inside the observational domain,
it is clearer there than in the wider domain we originally used.

**There is also a difference in what the two calculations measure.** Pellichero et al. do not
observe the surface fluxes. They solve the mixed layer heat and salt budgets with the flux as the
unknown, so what they recover is the surface flux together with the diapycnal mixing term, which
that budget cannot separate, whereas our model diagnoses the surface flux alone. Their stated
uncertainties are the mixed layer detection method, the instrumental errors in pressure,
temperature and conductivity, and the spatial sampling, which is sparsest under winter ice. Some
difference is therefore expected before any model bias enters. Their freshwater term, by contrast,
is the same quantity as ours, since they define it as the flux from evaporation, precipitation and
ice formation or melt, so that side of the comparison is directly commensurate.

**An independent reanalysis supports the diagnostic.** Because the reviewer's concern is whether
the model or the method is at fault, we also computed the transformation from ORAS5 over both
domains, removing the sea ice term from our own flux to match what the reanalysis can supply
(Fig. R2.1c-f). Over the sea ice zone in the annual mean the two agree closely, with densification
peaks of +5.0 Sv for us and +4.4 Sv for ORAS5 one density class apart. Over the paper's own domain
and season, south of 60°S in JJA, both are positive almost everywhere and peak in the same density
class, but ours is the stronger, +62.9 against +36.8 Sv, which we attribute to our winter mixed
layers being deeper and more extensive than in the reanalysis rather than to the diagnostic. We
present this as a consistency check rather than as validation, since ORAS5 is a model constrained
by observations and its surface fluxes are the fields least constrained by any observation."""

sub(OLD, NEW, 'R2 Major issue 1: rewritten from the R1 answer, Fig R1.1 copied in as R2.1')

# The old Figure R2.1 caption now belongs to the domain-sensitivity figure, which
# has become R2.2; the figures that followed shift by one.
sub("""**Figure R2.1 (manuscript Fig. 8).** Transformation integrated south of 60°S, the domain used in
the submitted version (top row), and over the seasonal sea ice zone alone, which reproduces the
domain of Pellichero et al. to within 2.5% in area (bottom row).""",
    """![](letter_figs/figR1_wmt_domain_sensitivity.png)

**Figure R2.2 (manuscript Fig. 8).** Transformation integrated south of 60°S, the domain used in
the submitted version (top row), and over the seasonal sea ice zone alone, which reproduces the
domain of Pellichero et al. to within 2.5% in area (bottom row).""",
    'renumber old R2.1 -> R2.2 and restore its image')

sub('**Figure R2.2 (new analysis).** Transformation in the seasonal sea ice zone, the',
    '**Figure R2.3 (new analysis).** Transformation in the seasonal sea ice zone, the',
    'renumber old R2.2 -> R2.3')

sub('**Figure R2.3 (new analysis).** Where the model actually convects.',
    '**Figure R2.4 (new analysis).** Where the model actually convects.',
    'renumber old R2.3 -> R2.4')

sub("""**Figure R2.4 (manuscript Fig. 6).** Global overturning streamfunction (top) and anomalies relative
to PI (bottom).""",
    """**Figure R2.5 (manuscript Fig. 6).** Global overturning streamfunction (top) and anomalies relative
to PI (bottom).""",
    'renumber old R2.4 -> R2.5')

P.write_text(s)
print('applied %d edits:' % len(edits))
for e in edits:
    print('  -', e)
