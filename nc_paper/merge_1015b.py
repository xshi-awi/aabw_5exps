#!/usr/bin/env python
"""
Second merge batch for revision/response_letter_R12_1015.docx:
his P22/P24/P25 rewrites of the G1 body, and the P26 expansion.

He changed the shape of the argument here. The earlier letter declined to
compute a reanalysis transformation at all; he now computes one from ORAS5 and
compares it, so the "not feasible, for two reasons" framing is replaced by what
ORAS5 can and cannot supply. The Pellichero rejection quote stays, because it
is what bounds the claim the comparison can support.
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


# ------------------------------------------------------- P22: he added the shading sentence
sub("""The two curves share their structure. Both show buoyancy gain at the lighter classes, a zero
crossing, and densification at the heavier classes, with the freshwater term carrying the light
end and both curves converging towards zero at the dense end. In our pre-industrial state the""",
    """As seen in Fig. R1.1a-b, the two curves share their structure. Both show buoyancy gain at the
lighter classes, a zero crossing, and densification at the heavier classes, with the freshwater
term carrying the light end and both curves converging towards zero at the dense end. Freshwater
component dominate the light water class (< 27.1 kg/m3) and the dense end (> 27.67 kg/m3), shown
by the blue shading in Fig. R1.1a, while the thermal signature is confined to a band at
intermediate water densities, the pink shading in Fig. R1.1a. In our pre-industrial state the""",
    'P22  shading sentence and figure callout')

# ---------------------------------------- P24: he split the definitional difference into its own para
sub("""Two differences should be noted, and they are of different kinds. Our light-end buoyancy gain is
weaker than theirs, and our whole curve sits at lighter densities, because the simulated Southern
Ocean surface is fresher than observed; both follow from the same bias, which we quantify under
model limitations below. The other difference is definitional rather than physical, and it sets a
floor on how closely the two can ever agree. What Pellichero et al. solve for is the surface flux
together with the diapycnal mixing term, because the mixed layer budget cannot separate them,
whereas our model diagnoses the surface flux alone. Some discrepancy is therefore expected before
any model bias is considered, and the agreement in shape should be read in that light.""",
    """Two differences should be noted, and they are of different kinds. Our light-end buoyancy gain is
weaker than theirs, and our whole curve sits at lighter densities, because the simulated Southern
Ocean surface is fresher than observed; both follow from the same bias, which we quantify under
model limitations below.

It should be noted that the difference could be definitional, and it sets a floor on how closely
the two can ever agree. Pellichero et al. solve the mixed layer heat and salt budgets for the
surface fluxes as the unknown, using the observed seasonal cycle of mixed layer temperature,
salinity and depth together with Ekman and geostrophic advection and entrainment. Thus, what
Pellichero et al. solve for is the surface flux together with the diapycnal mixing term, because
the mixed layer budget cannot separate them, whereas our model diagnoses the surface flux alone.
Some discrepancy is therefore expected before any model bias is considered, and the agreement in
shape should be read in that light. Their freshwater term, by contrast, is the same quantity as
ours, since they define it as the flux from evaporation, precipitation and ice formation or melt,
so the sea ice contribution is included on both sides of that particular comparison.""",
    'P24  definitional difference split out, freshwater term equivalence added')

# ------------------------- P25 + P26: ORAS5 is now computed rather than declined, and expanded
OLD_REAN = """The reviewer, and Reviewer 3, also asked whether we could compute an observational transformation
ourselves from a reanalysis rather than citing a published one. We examined this and it is not
feasible, for two reasons.

First, such data do not exist. A transformation budget needs the net surface heat flux, the net
surface freshwater flux, and the sea ice thermodynamic salt flux. We checked the Copernicus ocean
reanalyses, namely GLORYS12, the GREP ensemble and the analysis-forecast product, which carry
temperature, salinity, velocity and sea ice state but no surface flux fields at all. ORAS5 does
carry a net downward heat flux and a net upward water flux, but we tested its water flux inside
the September ice zone and found no freezing signal: at peak freezing it is -1.2e-5 kg m-2 s-1
against -1.6e-5 at peak melt, so the field is evaporation minus precipitation minus runoff and
excludes the sea ice term. The sea ice contribution is precisely the term that decides the
question at issue, and no reanalysis we could find supplies it.

Second, Pellichero et al. themselves reject that approach for this region. They write that ocean
buoyancy fluxes are commonly obtained from atmospheric reanalysis datasets, but that
*"in the Southern Ocean such reanalysis products are poorly constrained, and essentially unusable
in the seasonally sea-ice covered region."* Rather than use one, they solve the mixed layer heat
and salt budgets for the surface fluxes as the unknown, using the observed seasonal cycle of mixed
layer temperature, salinity and depth together with Ekman and geostrophic advection and
entrainment. The hydrography comes from Argo floats, ship-based observations and instrumented
marine mammals, the only platform that samples under winter sea ice; the ice edge comes from SMMR
and SSM/I passive microwave data for 2000-2015, and the geostrophic term from AVISO mean dynamic
topography for 1993-2012. We cannot reproduce that calculation, because it rests on a seal-tag
mixed layer database those authors assembled and we do not have. Computing a reanalysis
transformation anyway would mean doing the very thing their paper argues cannot be done in this
region.

We have therefore compared against their published result rather than recomputing one."""

NEW_REAN = """The reviewer, and Reviewer 3, also asked whether we could compute an observational transformation
ourselves from a reanalysis rather than citing a published one. A transformation budget needs the
net surface heat flux, the net surface freshwater flux, and the sea ice thermodynamic salt flux.
We checked the Copernicus ocean reanalyses, namely GLORYS12, the GREP ensemble and the
analysis-forecast product, which carry temperature, salinity, velocity and sea ice state but no
surface flux fields at all. ORAS5 does carry a net heat flux and a net water flux, but excluding
the sea ice term: we tested its water flux inside the September ice zone and found no freezing
signal, -1.2e-5 kg m-2 s-1 at peak freezing against -1.6e-5 at peak melt, so the field is
evaporation minus precipitation minus runoff. Therefore, we calculate transformation based on
ORAS5, and for consistency, we also remove the sea ice term from our model, then compare our model
results with ORAS5.

The results are shown in Fig. R1.1c-f. Over the sea ice zone in the annual mean, panels (c) and
(d), the agreement is close. Our densification branch peaks at +5.0 Sv and the ORAS5 branch at
+4.4 Sv, one density class apart at γ 27.49 against 27.28, and both show the light-end buoyancy
gain, ours reaching a minimum of -8.0 Sv against -12.9 Sv in the reanalysis. The sign structure,
the position of the zero crossing and the magnitude of the densification peak therefore agree on
a domain neither calculation was tuned to.

Over the domain and season the paper itself uses, south of 60°S in JJA, panels (e) and (f), the
two agree in shape but not in amplitude. Both are positive almost everywhere, since in winter the
whole sector is losing buoyancy, and both peak in the same density class at γ 27.28 and 27.08
respectively. Ours is the stronger, +62.9 Sv against +36.8 Sv. We do not attribute that gap to the
reanalysis being correct. Our winter mixed layers south of 60°S are deeper and more extensive than
in ORAS5, which exposes more water to the atmosphere and so supports more transformation, and the
same bias is visible directly in the mixed layer depth comparison shown later in this response. We
therefore read the winter comparison as agreement in sign, shape and density class, with an
amplitude difference that follows from a mixed layer bias we document rather than from the
diagnostic itself.

In the densest classes, above γ 27.7, neither curve sustains much transformation once the sea ice
term is removed from both sides, which is the expected outcome: that is the class where brine
rejection does the work, and it is exactly the term ORAS5 cannot supply. With the term restored,
our own dense-class transformation is positive, which is the result the paper rests on and the one
no available reanalysis can be used to check.

Two caveats bound what this comparison can support. ORAS5 is a model constrained by observations
rather than a measurement, and the two fields the calculation needs are the two that are least
constrained, since no observing system measures an air-sea flux directly; its heat flux is
computed from ERA-Interim forcing and then adjusted by sea surface temperature restoring, not
fitted to any flux observation. And Pellichero et al. themselves reject this approach for this
region, writing that ocean buoyancy fluxes are commonly obtained from atmospheric reanalysis
datasets but that *"in the Southern Ocean such reanalysis products are poorly constrained, and
essentially unusable in the seasonally sea-ice covered region."* We therefore present the ORAS5
comparison as a consistency check on the diagnostic rather than as independent validation, and we
continue to treat their published result as the observational reference."""

sub(OLD_REAN, NEW_REAN, 'P25/P26  ORAS5 now computed and compared, expansion of Fig R1.1c-f')

P.write_text(s)
print('applied %d edits:' % len(edits))
for e in edits:
    print('  -', e)
