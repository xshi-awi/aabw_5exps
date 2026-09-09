#!/usr/bin/env python
"""
Strengthen the SAM narrative in the main text with the new push/pull analysis,
while all SAM figures now live in the supplement.
"""
import sys
from pathlib import Path

P = Path('build/revised.tex')
s = P.read_text()
edits = []


def E(old, new, label):
    edits.append((old, new, label))


# Add the push/pull result as the substantive new SAM finding.
E(r"""Together, these results demonstrate that the Southern Annular Mode systematically modulates transformation rates in all five simulated climate states through two distinct contributors: thermal forcing, which dominates during interglacials via turbulent heat fluxes, and haline forcing, which dominates during glacials via sea ice processes.""",
  r"""These composites indicate that the mode acts on dense water formation through two routes at once, rather than through the wind alone. To separate them we computed, for the same high and low phases, the zonally integrated northward Ekman transport at 60°S and the coastal brine input integrated south of 65°S (Supplementary Fig. \ref{sam_pushpull}). The positive phase strengthens the Ekman transport by 8.0--12.8~Sv in every climate state, which is the wind-driven upwelling that supplies water to the surface. It simultaneously increases the coastal brine input, by 51 and 42~mSv in PI and LIG and by 24 and 31~mSv in LGM and MIS3, which is the salt pump that converts that water into denser classes. Across the five states the two responses scale together ($r = 0.77$), and the transformation response follows both.

The spatial patterns show why (Supplementary Fig. \ref{sam_windice}). Strengthened westerlies under the positive phase drive a sea ice dipole, with ice lost near the coast where polynyas open and brine is rejected, and gained further north where the exported ice melts. The dipole sits further north and is markedly stronger in the glacial states.

An asymmetry between the two routes emerges when the mean state changes. The Ekman response is largest in the interglacials, at 11.2~Sv in PI and 12.8~Sv in LIG, and weakest at the Last Glacial Maximum, at 8.0~Sv, even though the glacial mean wind stress is stronger. Extensive glacial ice cover transmits the wind stress to the ocean less efficiently and damps the divergence that the same wind anomaly would otherwise produce. The haline route is correspondingly more important in the glacial states, so the mode continues to modulate transformation by a comparable amount while operating through a different pathway.

Together, these results demonstrate that the Southern Annular Mode systematically modulates transformation rates in all five simulated climate states through two distinct contributors: thermal forcing, which dominates during interglacials via turbulent heat fluxes, and haline forcing, which dominates during glacials via sea ice processes.""",
  'sam-pushpull-results')

# Point the reader to the supplement where the SAM figures now live.
E(r"""Through composite analysis comparing high-SAM versus low-SAM years  (defined as JJA-mean SAM index exceeding $\pm$1.2$\sigma$ from the climatological mean)  using 100-year monthly simulations, we examine how this atmospheric forcing modulates   dense water formation across different climate states and regional sectors  (Fig. \ref{sam_wmt}).""",
  r"""Through composite analysis comparing high-SAM versus low-SAM years (defined as JJA-mean SAM index exceeding $\pm$1.2$\sigma$ from the climatological mean) using 100-year monthly simulations, we examine how this atmospheric forcing modulates dense water formation across different climate states and regional sectors. The supporting composite fields are collected in Supplementary Figs. \ref{sam_pushpull}--\ref{sam_wmt}.""",
  'sam-fig-pointer')

# Discussion: add the push/pull framing.
E(r"""The Southern Annular Mode emerges as a persistent modulator of transformation across all climate states. In its positive phase it enhances turbulent heat loss over open water and simultaneously drives sea ice divergence that promotes brine rejection near the coast, with composite anomalies of 3--8 Sv between high and low phases.""",
  r"""The Southern Annular Mode emerges as a persistent modulator of transformation across all climate states. In its positive phase it enhances turbulent heat loss over open water and simultaneously drives sea ice divergence that promotes brine rejection near the coast, with composite anomalies of 3--8 Sv between high and low phases. The two effects are not independent expressions of the same forcing but the two halves of a single mechanism. A positive anomaly strengthens the Ekman divergence that brings deep water to the surface, and in the same winters it strengthens the coastal salt pump that converts that water into denser classes, so the wind acts on both the supply of water to the surface and its conversion once there. Because these respond together across all five states, wind variability modulates dense water formation more effectively than either route alone would suggest.""",
  'sam-discussion-pushpull')

# Note the state dependence of the Ekman route in the Discussion.
E(r"""The robustness of this SAM-WMT coupling across dramatically different background conditions indicates that atmosphere-ocean-sea ice interactions operate through consistent physical mechanisms regardless of mean climate state.""",
  r"""The robustness of this coupling across dramatically different background conditions indicates that atmosphere-ocean-sea ice interactions operate through consistent physical mechanisms regardless of mean climate state, although the balance between the two routes shifts. The Ekman response to a given phase anomaly is weaker in the glacial states than in the interglacials despite a stronger mean wind stress, because extensive ice cover damps the transmission of stress to the ocean, while the haline route strengthens. The net modulation is of comparable magnitude in every state, but it is delivered through a different pathway, which mirrors the mean-state regime shift.""",
  'sam-discussion-asymmetry')

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
print(f'applied {len(edits)} SAM text edits')
