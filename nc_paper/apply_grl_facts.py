#!/usr/bin/env python
"""
Add the facts that are now verifiable from the PDFs the author supplied
(Shi et al. 2022 J. Climate; Shi et al. 2025 GRL and its supplementary figures).

What these documents DO support, and is used here:
  - the five-experiment ensemble of this manuscript is the same ensemble used in
    Shi et al. 2025 GRL, which states it explicitly
  - GRL Figure S3 shows Southern Hemisphere sea ice concentration for all five
    states, so the simulated glacial sea ice expansion is already published
  - GRL reports JJAS global mean surface air temperature for the five states
    (PI 15.5, MH 15.6, LIG 16.9, LGM 11.1, MIS3 13.0 degrees C), which sets the
    scale of the glacial-interglacial signal against which drift is judged
  - Shi et al. 2022 J. Climate states the 1000-year integration and the
    +/-0.05 K/century drift criterion verbatim

What they do NOT support, and is therefore not claimed:
  - the authoritative five-run experimental design lives in GRL Text S1, which is
    not part of the supplementary figures PDF, so no spin-up length is inherited
    from it. The manuscript keeps the author's own 1500/1000 figures.
"""
import sys
from pathlib import Path

P = Path('build/revised.tex')
s = P.read_text()
edits = []


def E(old, new, label):
    edits.append((old, new, label))


# The glacial-interglacial temperature contrast, from the GRL paper, gives the
# scale against which the drift should be judged. Replace my generic "order 4 K".
E(r"""The glacial-interglacial differences examined here are of order 4~K in global mean temperature, so a century-scale drift of 0.1~K is two orders of magnitude smaller than the signal and cannot account for it.""",
  r"""The glacial-interglacial differences examined here are large by comparison: the same five simulations give a JJAS global mean surface air temperature of 15.5, 15.6 and 16.9~$^{\circ}$C for PI, MH and LIG against 11.1 and 13.0~$^{\circ}$C for LGM and MIS3 \cite{Shi2025GRL}, a contrast of several kelvin. A century-scale drift of 0.1~K is therefore two orders of magnitude smaller than the signal and cannot account for it.""",
  'drift-scale')

# The simulated sea ice of this ensemble is already published, which strengthens
# the answer to Reviewer 2's question about how the model achieves glacial ice.
E(r"""The simulated glacial sea ice of this model has been assessed against diatom-based reconstructions in a multi-model comparison, which found that models including AWI-ESM tend to underestimate the winter sea ice extent inferred from the proxies while capturing the summer distribution more closely \cite{Green2022}.""",
  r"""The Southern Hemisphere sea ice distribution of these same five simulations has been published previously \cite{Shi2025GRL}, and the pronounced glacial expansion described here is evident there as well. The simulated glacial sea ice of this model has also been assessed against diatom-based reconstructions in a multi-model comparison, which found that models including AWI-ESM tend to underestimate the winter sea ice extent inferred from the proxies while capturing the summer distribution more closely \cite{Green2022}.""",
  'seaice-published')

# Make explicit that this is the same ensemble, which is useful provenance.
E(r"""These simulations have been documented and evaluated against proxy data in previous studies with this model.""",
  r"""These simulations have been documented and evaluated against proxy data in previous studies with this model, and the five-member ensemble analysed here is the same one used in those studies.""",
  'same-ensemble')

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
print(f'applied {len(edits)} edits from the supplied PDFs')
