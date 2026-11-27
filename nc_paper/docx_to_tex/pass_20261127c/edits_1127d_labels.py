#!/usr/bin/env python3
"""11-27 10:34 user note (via questions-ef): smaller lat/lon labels on every panel of R4 (Fig. 6), R7 (Fig. 3/4)
and R16 (A10); R9/R10 (A7/A8) without the sea ice contour and with lat/lon labels; R20 (A2) labelled and
only panels a-e in the letter. Captions and the R1 map-label reply follow the figures.
Usage: edits_1127d_labels.py revised <revised.tex>   or   edits_1127d_labels.py letter <response_new.tex>"""
import sys

mode, p = sys.argv[1], sys.argv[2]
s = open(p, encoding='utf-8').read()
T = '%claude回复（11-27 11:00）：'


def rep(old, new, n=1):
    global s
    c = s.count(old)
    assert c == n, (c, old[:90])
    s = s.replace(old, new)


if mode == 'revised':
    # A7 and A8: the 15% ice edge is no longer drawn
    rep(r' The black line is the 15\% winter sea ice edge of each state.', '', n=2)
elif mode == 'letter':
    rep(r' The black line is the 15\% winter sea ice edge of each state.', '')
    rep(r'''\rfig{0.9}{../letter_figs/fig02_mld1000.png}{Revised Supplementary Fig. A2. June-September mean mixed layer depth (a-e) and sea ice concentration (f-j). The mixed layer depth colour scale now runs to 1000 m in the interglacial panels (a-c) and stays at 400 m in the glacial panels (d,e).\label{fR:mldA2}}''',
        r'''\rfig{0.9}{../letter_figs/fig02_mld1000_mldonly.png}{Revised Supplementary Fig. A2, mixed layer depth panels (a-e). June-September mean mixed layer depth. The colour scale now runs to 1000 m in the interglacial panels (a-c) and stays at 400 m in the glacial panels (d,e).\label{fR:mldA2}}
''' + T + r'''按你 10:34 的意思，信里这张只放 MLD 的 a--e，排成两行（a--c 一行、d--e 一行，各配原来的色标），每格加了经纬度（约 6 pt），地图因此印出约 1.6 in 见方。补充材料里的 A2 没动，仍是 MLD 加海冰两行、没有经纬度标注。稿件 A2 要不要也去掉海冰行、加经纬度，你定。''')
    # R1 map labels: describe what the figures now show
    rep(r'''On the map labels: the polar stereographic panels now carry labelled latitude circles, and the ventilation age maps carry labelled meridians as well. On the fifteen-panel composite figures, we label the latitude circles only, because a full graticule with meridian labels on panels of that size was illegible in trial versions and crowded the panel margins. We are happy to add complete labelling if the reviewer prefers, or to enlarge those figures to accommodate it. As a concrete example, the ventilation age maps now carry labelled latitude circles and labelled meridians, with 180° on every panel and 60°W, 120°W, 60°E and 120°E on the outer panels of each row, and the mixed layer and composite figures carry labelled latitude circles. Figure~\ref{fR:agelab} in this letter shows the result. The same treatment has been applied to every map figure in the revised manuscript and supplement.''',
        r'''On the map labels: the polar stereographic panels now carry labelled latitude circles and labelled meridians. In the multi-panel map figures, such as the surface density tendency maps (Figs. 3 and 4), the ventilation age maps (Fig. 6) and the SAM composites (Supplementary Figs. A7, A8 and A10), every panel carries these labels, set in a smaller font than the rest of the figure so that they hide as little of the data as possible. Figure~\ref{fR:agelab} in this letter shows the ventilation age maps as an example.''')
    rep(r'Revised manuscript figure. The ventilation age figure as an example of the labelling now applied throughout.',
        r'Revised manuscript Fig. 6. The ventilation age maps as an example of the labelling.')
    rep(r'''Every panel carries labelled latitude circles and labelled meridians, and the climate state is named above each panel. The mixed layer, wind stress and composite figures have been treated the same way; please refer to the revised manuscript and supplement for the complete set.\label{fR:agelab}}''',
        r'''Every panel carries labelled latitude circles and labelled meridians, and the climate state is named above each panel. Figs. 3 and 4 and Supplementary Figs. A7, A8 and A10 are labelled in the same way.\label{fR:agelab}}
''' + T + r'''你 10:34 说经纬度太大、每条纬线都要标：R4、R7（及 Fig. 4）、R16 现在每格都标全经线和纬线，字约 6 pt；R9、R10 去掉了海冰线（图注那句删了，稿件 A7、A8 图注同删），也加了同样的经纬度。上面回复里原来说「15 格的合成图只标纬度」「每张地图都同样处理」，现在都不准了，改成只列确实每格标全的几张图（A2、A12、A4、A13 的标注不全，所以没写「全部」）。''')
else:
    sys.exit('mode must be revised or letter')

assert '\t' not in s and '\r' not in s
open(p, 'w', encoding='utf-8').write(s)
print('ok', mode, p)
