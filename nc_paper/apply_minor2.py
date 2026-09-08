#!/usr/bin/env python
"""Third pass: items promised in the response letter that were still outstanding."""
import sys
from pathlib import Path

P = Path('build/revised.tex')
s = P.read_text()
edits = []


def E(old, new, label):
    edits.append((old, new, label))


# R1#16: "see Section 2" -> name the Methods section explicitly
E(r"""we convert heat and freshwater fluxes into surface density tendency during austral winter (see Section 2 for more details).""",
  r"""we convert heat and freshwater fluxes into surface density tendency during austral winter (see Methods for more details).""",
  'section-ref')

# R2 and R3: define the experiment abbreviations at first use in the Results
E(r"""\subsection{Large-scale features of the Southern Ocean during austral winter}""",
  r"""\subsection{Large-scale features of the Southern Ocean during austral winter}

Throughout the Results we refer to the five simulated climate states by the abbreviations used in the figures: PI for pre-industrial, MH for mid-Holocene, LIG for the last interglacial, LGM for the Last Glacial Maximum, and MIS3 for Marine Isotope Stage 3. Their boundary conditions are summarised in Table \ref{tab:bc}.""",
  'acronyms-results')

# R3 L129: note that the glacial salinity anomaly contains the uniform global offset
E(r"""Here the enhanced glacial salinification partly reflects our model initialization with higher global mean salinity, accounting for ice sheet storage of freshwater.""",
  r"""Here the enhanced glacial salinification partly reflects our model initialization with higher global mean salinity, accounting for ice sheet storage of freshwater. Because that offset is applied uniformly, the anomaly maps in Fig. \ref{sst}g,h combine it with the locally generated signal, and the regional structure rather than the absolute magnitude is what reflects local dynamics.""",
  'salinity-offset-note')

# R1#13: Ross Sea sea ice / wind stress explanation
E(r"""This intensified wind stress enhances Ekman transport and facilitates the northward expansion of sea ice, fundamentally altering the air-sea interaction interface.""",
  r"""This intensified wind stress enhances Ekman transport and facilitates the northward expansion of sea ice, fundamentally altering the air-sea interaction interface. The response is not uniform around the continent. In the Ross sector the stronger stress drives enhanced northward ice export, but the surface ocean there remains comparatively warm, so replacement ice growth is limited thermodynamically rather than dynamically. Brine rejection strengthens without a proportional increase in ice area, which is also why this sector departs from the circumpolar pattern in the variability analysis below.""",
  'ross-sea-explanation')

# R2: polynya resolution and how brine enters the ocean
E(r"""In our configuration, the grid resolution ranges from $\sim$100 km in the open ocean to 25 km in polar regions and along coastlines, with 46 distinct vertical layers.""",
  r"""In our configuration, the grid resolution ranges from $\sim$100 km in the open ocean to 25 km in polar regions and along coastlines, with 46 distinct vertical layers. At this resolution the larger coastal polynyas are represented as features, but the boundary layer processes that set polynya dynamics are not resolved. Sea ice growth and melt are computed thermodynamically at each surface node, and the associated brine release enters the ocean as a salt flux at that node, so the brine signal appears wherever the model forms ice; its intensity, however, depends on the simulated wind-driven ice divergence rather than on a resolved polynya circulation. The configuration does not include ice shelf cavities, so the ocean beneath floating ice shelves, and the basal melt it supplies, are absent from the simulations. Regions covered by floating ice shelves are therefore outside the model domain and appear blank in the map figures.""",
  'polynya-resolution')

# R3 L206-209: move the general statement to the Introduction.
# In the submitted text this is the sentence about sea ice acting as a freshwater pump.
E(r"""Fig. \ref{density_fwf}f exhibits a classic dipole pattern in the sea ice-driven density tendency.""",
  r"""The sea ice contribution shows a dipole in space. Fig. \ref{density_fwf}f exhibits this classic pattern in the sea ice-driven density tendency.""",
  'fwf-dipole-lead')

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
print(f'applied {len(edits)} edits')
