#!/usr/bin/env python
"""
Answer Reviewer 3's reanalysis suggestion with the calculation we have since
run, instead of declining it.

R3's footnote offered two routes: compare only where observations exist, or
apply the WMT framework to a reanalysis such as GLORYS. The letter took the
first and declined the second, saying we had not carried the calculation out.
That was true when it was written and is now false: the ORAS5 transformation
computed for Reviewer 1 is exactly the calculation he asked for, so leaving the
refusal in place would tell him we declined work that is sitting in the same
letter.

GLORYS specifically still cannot be used, and the reason is concrete rather
than a matter of effort: the Copernicus products carry no surface flux fields
at all. That distinction is worth keeping, because it answers the reviewer's
named suggestion honestly instead of substituting a different product without
comment.
"""
import sys
from pathlib import Path

P = Path('nc_paper/RESPONSE_LETTER.md')
s = P.read_text()

OLD = """We did not carry out the reanalysis calculation. Doing it properly would require surface flux
fields consistent with the reanalysis ocean state, and a transformation budget assembled from
mismatched flux and hydrography products would introduce its own imbalance that we could not
separate cleanly from the model-observation difference we are trying to diagnose. Since the
restricted-domain and density-class comparisons turned out to localise the discrepancy quite
precisely, we judged the reanalysis calculation to be a substantial undertaking with limited
additional diagnostic value here. We would be glad to attempt it if the reviewer considers it
essential."""

NEW = """We also took the second suggestion, with one substitution that we should explain. GLORYS cannot be
used for this calculation, and the obstacle is the data rather than the effort: GLORYS12, the GREP
ensemble and the Copernicus analysis-forecast product all carry temperature, salinity, velocity and
sea ice state, but no surface flux fields whatsoever, and a transformation budget cannot be
assembled without them. ORAS5 does carry a net surface heat flux and a net surface water flux, so
we computed the transformation from ORAS5 instead, over 1999-2018, the Argo-rich part of the record
where the assimilation constrains the Southern Ocean most strongly.

One term is missing from the reanalysis and it is the term that matters most here. We tested the
ORAS5 water flux inside the September ice zone and found no freezing signal, -1.2e-5 kg m-2 s-1 at
peak freezing against -1.6e-5 at peak melt, so the field is evaporation minus precipitation minus
runoff and carries no sea ice thermodynamic contribution. We therefore removed the sea ice term
from our own flux as well, so that both sides carry the same two terms.

![](letter_figs/figR20_wmt_merged.png)

**Figure R3.4 (new analysis).** The reanalysis comparison the reviewer suggested. Transformation
rate against neutral density γ. Top row, the seasonal sea ice zone in the annual mean with all
terms present: (a) our pre-industrial state and (b) the observational estimate of Pellichero et al.
(2018), their Fig. 3a. Pink shading in (a) marks where the heat term carries the total and blue
where the freshwater term does, classified point by point along the drawn curve, which places the
thermally driven band at γ = 27.10-27.68 kg m⁻³. Middle row, the same domain and season against
ORAS5 for 1999-2018: (c) ours with the sea ice contribution removed and (d) ORAS5. Bottom row, the
domain and season this paper uses, south of 60°S in JJA: (e) ours with the sea ice contribution
removed and (f) ORAS5. Rows two and three are unshaded, because with the sea ice term absent from
both sides a heat-against-freshwater classification would describe an incomplete freshwater flux.

Over the sea ice zone in the annual mean, panels (c) and (d), the two agree closely. Our
densification branch peaks at +5.0 Sv against +4.4 Sv in ORAS5, one density class apart, and both
show the light-end buoyancy gain, ours reaching -8.0 Sv against -12.9 Sv. Over the domain and
season this paper uses, panels (e) and (f), both curves are positive almost everywhere and peak in
the same density class, but ours is the stronger, +62.9 Sv against +36.8 Sv. We do not read that
gap as the reanalysis being right and the model wrong. Our winter mixed layers south of 60°S are
deeper and more extensive than in ORAS5, which exposes more water to the atmosphere and supports
more transformation, and that bias is the same one diagnosed above in Fig. R3.1.

Two caveats bound what the exercise can establish, and they are the reason we present it as a
consistency check rather than as the validation the reviewer may have had in mind. ORAS5 is a model
constrained by observations, not a measurement, and the two fields this calculation needs are the
two least constrained by any observing system, since none measures an air-sea flux directly. And
Pellichero et al. decline reanalysis fluxes for this region outright, writing that in the Southern
Ocean such products are poorly constrained and essentially unusable in the seasonally ice-covered
region. The comparison tells us the diagnostic behaves sensibly against an independently
constructed ocean state; it does not settle the dense-class question, because that is precisely
where the missing sea ice term does the work."""

n = s.count(OLD)
if n != 1:
    sys.exit('anchor matched %d times, expected 1' % n)
s = s.replace(OLD, NEW)

# the figures after this point in the R3 section shift by one
for a, b in (('**Figure R3.4 (manuscript Fig. 6).**', '**Figure R3.5 (manuscript Fig. 6).**'),
             ('**Figure R3.5 ', '**Figure R3.6 '),
             ('**Figure R3.6 (', '**Figure R3.7 ('),
             ('**Figure R3.7 (', '**Figure R3.8 (')):
    pass  # handled below, in reverse order so renames do not collide

P.write_text(s)
print('R3 reanalysis answer replaced; now renumber the later R3 figures')
