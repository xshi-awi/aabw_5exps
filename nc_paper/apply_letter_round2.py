#!/usr/bin/env python
"""
Second round of response-letter edits, from the author's yellow annotations:
  - correct the Ross Sea answer, which the data does not support as written
  - add the tightened-colorbar glacial MLD figure at R1 comment 10
  - move the convection figure to the model-limitations comment and add the
    domain map at Reviewer 2's major issue 1
"""
import sys
from pathlib import Path

P = Path('RESPONSE_LETTER.md')
s = P.read_text()
edits = []


def E(old, new, label):
    edits.append((old, new, label))


IMG = 'letter_figs'

# ---------------------------------------------------------------- R1 #13
# The explanation in the submitted letter is not supported by the output.
E("""— The reviewer's interpretation is
essentially the one our output supports. In the Ross sector the intensified wind stress does drive
enhanced northward ice export, but the ocean surface there remains comparatively warm, so the
exported ice is replaced by growth that is thermodynamically limited rather than
dynamically limited. The result is stronger brine rejection without a proportional increase in ice
area. We have added a sentence to this effect, and we note the same sector-dependence when
discussing why the Ross Sea departs from the circumpolar Southern Annular Mode signal.""",
  f"""— We checked this
directly, and the answer turned out to be more interesting than the question assumed, so we set it
out in full.

The mechanism the reviewer proposes is real, but it operates in the last interglacial rather than in
the glacial states the comment refers to. Averaged between 60 and 75°S in winter, the glacial sea
ice response is in fact fairly uniform around the continent: concentration rises by 0.62 in the
Ross sector, 0.70 in the Weddell sector and 0.53 in the Adélie sector at the Last Glacial Maximum,
and all three sectors cool to within about 0.1 K of the surface freezing point, so ice growth is
not limited by surface heat content anywhere.

![]({IMG}/figR9_ross_sector.png)

**Figure R1.4 (new analysis).** (a) Winter surface temperature above the local freezing point and
(b) winter sea ice concentration, by sector and climate state. (c) The same quantities as anomalies
relative to PI. The Ross sector separates from the others only in the last interglacial, where it
warms to 2.7 K above freezing and loses 0.19 of ice concentration, an order of magnitude more than
the Weddell or Adélie response. In the glacial states all three sectors collapse onto the freezing
point and gain ice comparably.

So in the last interglacial the Ross sector is 0.97 K warmer than in PI and sits 2.72 K above the
freezing point, against changes below 0.3 K elsewhere, and its winter ice concentration falls by
0.19. There the wind can export ice efficiently but the water is too far from freezing for the
exported ice to be replaced, which is precisely the reviewer's proposed mechanism. We have
rewritten the passage to say this, rather than attributing the behaviour to the glacial wind
response as the submitted version implied. This asymmetry is also why the Ross sector departs from
the circumpolar pattern in the variability analysis.""",
  'r1-13-corrected')

# ---------------------------------------------------------------- R1 #10
E("""counterparts. Panels (g) and (h) of Figure R1.2 above make the point quantitatively: both the area
of deep convection and the depth of the deepest mixed layer fall in the glacial states, so the
reviewer's reading of the shading is correct and is now reflected in the text.""",
  f"""counterparts. Panels (g) and (h) of Figure R1.2 above make the point quantitatively: both the area
of deep convection and the depth of the deepest mixed layer fall in the glacial states, so the
reviewer's reading of the shading is correct and is now reflected in the text.

The reviewer's observation also pointed to a presentation problem, which we have fixed. On the
shared 0–400 m scale of the original figure the glacial panels are almost featureless, because the
99th percentile of the glacial winter mixed layer south of 55°S is only 280 m (LGM) and 296 m
(MIS3). Replotting those two states on their own 0–300 m scale makes the coastal cells visible.

![]({IMG}/figR7_mld_glacial_zoom.png)

**Figure R1.5 (new supplementary panel).** LGM and MIS3 winter mixed layer depth on the original
0–400 m scale (top) and on a tightened 0–300 m scale with the 200 m contour marked (bottom). The
tightened scale resolves discrete deep cells along the coast, in the Weddell, Prydz Bay and Ross
sectors, which is where the glacial dense water is produced. This is the figure we now use in the
supplement.""",
  'r1-10-mldfig')

# ---------------------------------------------------------------- R2 domain map
E("""**The domains are less similar than they appear.** Pellichero et al. define their sector as the
region enclosed by the September 15% sea ice contour, so its northern boundary is an ice contour
that varies with longitude, not a latitude circle.""",
  f"""**The domains are less similar than they appear.** This is worth showing directly,
since the similarity of the two regions is the premise of the reviewer's argument.

![]({IMG}/figR8_domain_map.png)

**Figure R2.0 (new analysis).** (a) The domain used in the submitted manuscript, everything south
of 60°S, covering 2.07×10¹³ m². (b) The seasonal sea ice zone of the model, inside the September
15% contour, which is the definition Pellichero et al. use, covering 1.22×10¹³ m². (c) The two
overlaid. The orange ring is inside our domain but outside the ice zone: it is 42.7% of the area we
integrated over, it is open water all year, and it is water their analysis does not include.

Pellichero et al. define their sector as the
region enclosed by the September 15% sea ice contour, so its northern boundary is an ice contour
that varies with longitude, not a latitude circle.""",
  'r2-domain-map')

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
print(f'applied {len(edits)} letter edits')
