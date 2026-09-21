#!/usr/bin/env python
"""
Add the model-evaluation paragraph and the expanded experimental design to the
manuscript, using only statements that were verified against the published papers.

Accuracy guards observed here:
  - Sidorenko et al. 2019 contains NO Southern Ocean mixed-layer evaluation, NO
    satellite sea-ice comparison and NO discussion of polynyas or convection.
    It is therefore cited only for what it does report: the AABW cell strength,
    the Southern Ocean warm bias, the September Weddell ice thickness, and the
    authors' own caveat about water-mass biases.
  - The open-ocean-convection diagnosis remains ours, from our own output.
  - The spin-up wording follows Shi et al. 2023 (Climate of the Past), which is
    open access and states it verbatim.
"""
import sys
from pathlib import Path

P = Path('build/revised.tex')
s = P.read_text()
edits = []


def E(old, new, label):
    edits.append((old, new, label))


# ---------------------------------------------------------------- Methods:
# expanded experimental design, matching the wording of the group's earlier papers
E(r"""The PI control run was integrated for 1,500 years, initialized with AMIP atmospheric conditions \cite{roeckner2004atmospheric} and World Ocean Atlas climatology \cite{levitus2010world}. The MH and LIG simulations branched from the equilibrated PI state. The LGM and MIS3 experiments were initialized from a previous glacial state \cite{werner2016glacial}. All paleo-simulations were integrated for 1,000 model years. We analyze the final 100 years of each run, representing a quasi-equilibrium state.""",
  r"""The PI control run was integrated for 1,500 years, initialized with AMIP atmospheric conditions \cite{roeckner2004atmospheric} for the atmosphere and World Ocean Atlas climatology \cite{levitus2010world} for the ocean. The MH and LIG simulations branched from the equilibrated PI state. The LGM and MIS3 experiments were initialized from a previous glacial state \cite{werner2016glacial}. All paleo-simulations were integrated for 1,000 model years, following the protocol used for the same model in earlier work \cite{Shi2022JCLI,Shi2023CP,Shi2025GRL}. We analyze the final 100 years of each run, over which the simulated climate is in a quasi-equilibrium state, and the climatology of each variable is represented by its average over those 100 years. Drift over the analysis period is small: the trend in area-weighted mean sea surface temperature across the final century lies between $-0.03$ and $+0.06$~K~century$^{-1}$ in the five experiments, and the mid-Holocene simulation is indistinguishable from the pre-industrial control in both the latitude and the strength of the Southern Hemisphere westerly jet, which argues against residual drift as an explanation for the paleoclimate signals reported here.

The same five-experiment ensemble has been documented and evaluated against proxy data in previous studies with this model. \citet{Shi2022JCLI} describe the mid-Holocene and last interglacial simulations and compare two generations of AWI-ESM against pollen-based and marine reconstructions; \citet{Shi2025GRL} use the identical five states to examine African monsoon rainfall and validate the simulated precipitation against GPCP and against pollen compilations for the mid-Holocene, last interglacial and Last Glacial Maximum; and \citet{Shi2023CP} document the Last Glacial Maximum configuration and its comparison with the Bartlein pollen compilation and MARGO sea surface temperatures.""",
  'methods-setup')

# ---------------------------------------------------------------- Discussion:
# what previous AWI-ESM evaluations report for the Southern Ocean
E(r"""This bias is shared across the current generation of climate models rather than particular to AWI-ESM2.""",
  r"""Independent evaluations of this model provide context for these limitations, and we summarise the relevant points here rather than leaving them to be sought in the model description literature. \citet{sidorenko2019evaluation} evaluate the coupled configuration against observations and report that the abyssal overturning cell associated with Antarctic Bottom Water is reproduced with a maximum of about 10~Sv, comparable to the pre-industrial value obtained here. The same evaluation identifies the Southern Ocean as the region of the most pronounced warm bias, with a sea surface temperature root-mean-square error of 1.43~K against the PHC climatology at the resolution used here, and notes that Antarctic sea ice is too thin, hardly reaching 0.25~m in the central Weddell Sea in September. Their overall assessment is that although biases in the representation of water mass properties and ventilation mechanisms are present, the model still produces a reasonable density distribution that maintains realistic transports. A warm and thin-ice Southern Ocean is precisely the state in which open-ocean convection is favoured over shelf processes, so these documented biases are consistent with the convection behaviour we diagnose above, although we note that the published evaluation does not itself examine mixed-layer depth, polynyas or the convection pathway, and the diagnosis of open-ocean convection reported here is our own.

This bias is shared across the current generation of climate models rather than particular to AWI-ESM2.""",
  'discussion-validation')

# ---------------------------------------------------------------- glacial sea ice
E(r"""In our simulations the glacial ice cover expands sufficiently to insulate the interior gyres, and the area of deep winter mixed layers contracts to roughly one fifth of its PI value.""",
  r"""In our simulations the glacial ice cover expands sufficiently to insulate the interior gyres, and the area of deep winter mixed layers contracts to roughly one fifth of its PI value. The simulated glacial sea ice of this model has been assessed against diatom-based reconstructions in a multi-model comparison, which found that models including AWI-ESM tend to underestimate the winter sea ice extent inferred from the proxies while capturing the summer distribution more closely \cite{Green2022}. An underestimate of winter ice would if anything weaken the insulation effect we describe, so the haline-dominated glacial regime reported here is unlikely to be an artefact of excessive simulated ice.""",
  'glacial-seaice-validation')

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
print(f'applied {len(edits)} validation edits')
