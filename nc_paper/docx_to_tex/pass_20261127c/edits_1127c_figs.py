#!/usr/bin/env python3
"""11-27 figure pass (text enlarged for an A4 printout of the letter): caption and width fixes.
Usage: edits_1127c_figs.py revised <revised.tex>   or   edits_1127c_figs.py letter <response_new.tex>
The 200 m orange contour of A4 / letter Fig. R5 (figR7_mld_glacial_zoom) was removed from the figure on
2026-10-14 at the user's request (git 8ffe0d5, script docstring); both captions still described it.
A4 is now inserted at full text width, so its labels print at about 7 pt in the manuscript as well."""
import sys

mode, p = sys.argv[1], sys.argv[2]
s = open(p, encoding='utf-8').read()


def rep(old, new):
    global s
    c = s.count(old)
    assert c == 1, (c, old[:90])
    s = s.replace(old, new)


if mode == 'revised':
    rep(r'\centering\includegraphics[width=0.8\textwidth]{figures/figR7_mld_glacial_zoom.pdf}',
        r'\centering\includegraphics[width=\textwidth]{figures/figR7_mld_glacial_zoom.pdf}')
    rep(r'(a) LGM and (b) MIS3, on a 0--300 m colour scale with the 200 m contour in orange. Units: m.',
        r'(a) LGM and (b) MIS3, on a 0--300 m colour scale. Units: m.')
elif mode == 'letter':
    rep(r'(a) LGM and (b) MIS3 winter mixed layer depth south of 60°S, on a 0–300 m scale with the 200 m contour in orange. The tightened scale',
        r'(a) LGM and (b) MIS3 winter mixed layer depth south of 60°S, on a 0–300 m scale. The tightened scale')
    # R1 map labels: the reply listed 60°W, 0°, 60°E and 120°E, but 0° was never labelled; since the
    # enlargement the age maps label 180° on every panel and 60°/120° W and E on the outer panels of each row
    rep(r'As a concrete example, the ventilation age maps now carry labelled meridians at 60°W, 0°, 60°E and 120°E as well as labelled latitude circles, and the mixed layer and composite figures carry labelled latitude circles.',
        r'As a concrete example, the ventilation age maps now carry labelled latitude circles and labelled meridians, with 180° on every panel and 60°W, 120°W, 60°E and 120°E on the outer panels of each row, and the mixed layer and composite figures carry labelled latitude circles.')
else:
    sys.exit('mode must be revised or letter')

assert '\t' not in s and '\r' not in s
open(p, 'w', encoding='utf-8').write(s)
print('ok', mode, p)
