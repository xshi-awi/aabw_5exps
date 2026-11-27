#!/usr/bin/env python3
"""11-27 pass on build/revised.tex: follow the user's 11-27 edits of the letter quotes and align
the manuscript with the Reviewer 3 replies. Usage: edits_1127_revised.py <revised.tex>
Every anchor must match exactly once, so the script refuses to run twice."""
import sys

p = sys.argv[1]
s = open(p, encoding='utf-8').read()


def rep(old, new, n=1):
    global s
    c = s.count(old)
    assert c == n, (c, old[:90])
    s = s.replace(old, new)


# abstract: first sentence of the SAM part removed by the user (letter, R1 comments 2 and 3);
# state names capitalised as elsewhere (R1 comment 1)
rep('The balance between these two pathways is set by sea ice extent, which is modulated by '
    'internal atmospheric variability. Further analysis suggests', 'Further analysis suggests')
rep('mid-Holocene, last interglacial, last glacial maximum, and marine isotope stage 3, we identify a '
    'sea-ice-mediated', 'mid-Holocene, Last Interglacial, Last Glacial Maximum, and Marine Isotope '
    'Stage 3, we identify a sea-ice-mediated')

# Section 2.1
rep('In the last interglacial the Ross sector stands out.', 'In the Last Interglacial the Ross sector stands out.')
rep('strengthen in both the glacial and the last interglacial states', 'strengthen in both the glacial and the Last Interglacial states')
rep('For the last interglacial the simulated Northern Hemisphere', 'For the Last Interglacial the simulated Northern Hemisphere')
rep('holding the last interglacial ice sheet orography', 'holding the Last Interglacial ice sheet orography')
# user cut this sentence in the R3 L134 quote
rep('the model produces deep mixed layers ($>$400 m) in the Weddell and Ross Sea gyres',
    'the model produces deep mixed layers in the Weddell and Ross Sea gyres')
rep('In PI this convection is concentrated in the Weddell sector, which accounts for 70\\% of the area '
    'with a winter mixed layer deeper than 400 m south of 55°S and for 97\\% of the area deeper than '
    '600 m, centred near 68°S in the interior of the gyre, away from the continental shelf.',
    'In PI this convection is concentrated in the Weddell sector.')
# the ~1 degree LIG retreat is the circumpolar mean; the Ross sector retreats ~5 degrees (same section)
rep('and the temperature gradient aloft steepens, while the sea ice edge retreats by about 1°.',
    'and the temperature gradient aloft steepens, while the circumpolar mean sea ice edge retreats by about 1°.')

# Section 2.3, user cut the end of the R3 L195 quote
rep('returns to the ocean as liquid runoff at the coast (Methods), and the freshwater budget we '
    'decompose is dominated by sea ice thermodynamics.',
    'returns to the ocean as liquid runoff at the coast (Methods).')

# Section 2.4
rep('(+4 Sv for LGM at $\\sigma_2 = 37.5$~kg/m$^3$; +4.2 Sv for MIS at',
    '(+4 Sv for LGM at $\\sigma_2 = 37.5$~kg/m$^3$; +4.2 Sv for MIS3 at')
rep('can reach about 8 Sv, occuring at', 'can reach about 8 Sv, occurring at')
rep('acts on dense water formation through two routes at once.', 'acts on dense water formation through two routes.')
rep('because the winter pack there stays close to full cover. The extra brine release near the coast '
    'therefore shows up in the freshwater flux (Supplementary Fig. \\ref{sam_pushpull}b) and leaves '
    'little trace in the ice concentration. In the glacial states',
    'because the winter ice there stays close to full cover. In the glacial states')
# user: delete this paragraph, the correlation is no longer discussed
rep('The size of the two responses differs between the climate states, and it does not separate the '
    'interglacial from the glacial states. The Ekman response is 11.2, 8.9 and 12.8~Sv in PI, the '
    'mid-Holocene and the Last Interglacial and 8.0 and 10.1~Sv in LGM and MIS3 (Supplementary Fig. '
    '\\ref{sam_pushpull}a). The brine response ranks the states in a similar order (Supplementary Fig. '
    '\\ref{sam_pushpull}b), which gives the correlation noted above. What changes between interglacial '
    'and glacial states is the local balance of the anomalous forcing, which we describe next.\n\n', '')
rep('Near the coast the concentration hardly changes because the winter pack stays close to full cover there.',
    'Near the coast the concentration hardly changes because the winter ice stays close to full cover there.')

# Discussion, formation location paragraph as the user edited its quote (R3 first major comment)
rep('The model therefore does form dense water near the coast, mostly where the winter pack ice is open. '
    'In the interglacial states this coastal water is denser and much smaller in volume than the water '
    'made offshore, and the simulated PI bottom water has the density of the offshore water. In the '
    'glacial states the class of the transformation maximum forms under sea ice offshore (89--91\\%), and '
    'the densest classes form on the shelf (53--59\\%) and under the ice further offshore. The glacial '
    'bottom water is denser than 99\\% of the winter surface water, and the 0.7--1.0~Sv that reach its '
    'density all form on the shelf. Because',
    'The model therefore does form dense water near the coast. In the interglacial states this coastal '
    'water is denser and much smaller in volume than the water made offshore. In the glacial states the '
    'class of the transformation maximum forms under sea ice offshore (89--91\\%), and the densest classes '
    'form both on the shelf (53--59\\%) and under the ice further offshore. Because')
rep('production of near-freezing shelf water are absent by construction.', 'production of near-freezing shelf water are absent.')
rep('the open water convection that the interglacial integrals include.', 'the open water convection that the interglacial periods include.')
# the user removed this claim from the R3 push/pull reply
rep('so the wind acts on both the supply of water to the surface and its conversion once there. Because '
    'these respond together across all five states, wind variability modulates dense water formation '
    'more effectively than either route alone would suggest. This circumpolar',
    'so the wind acts on both the supply of water to the surface and its conversion once there. This circumpolar')

open(p, 'w', encoding='utf-8').write(s)
print('ok', p)
