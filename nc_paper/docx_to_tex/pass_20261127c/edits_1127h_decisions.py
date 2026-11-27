#!/usr/bin/env python3
"""11-27 12:44 user decisions after the 0.1-bin switch (via questions-ef):
2.2 shares as sea ice plus other freshwater 10-37% (heat 63-90%, |term|/sum of |terms| at the JJA peak, Fig. 2);
densest classes as class means (0.44 vs 0.04 Sv, annual, 100-year data, both domains give the same);
Fig. 8 from 100 years of monthly output (calc_wmt_siz_100years.py) and the 'roughly doubles' sentence rewritten;
SAM numbers from the 0.1 composites (largest anomaly, as A9c); notes of edits_1127g removed.
Usage: edits_1127h_decisions.py revised <revised.tex> | letter <response_new.tex>"""
import re
import sys

mode, p = sys.argv[1], sys.argv[2]
s = open(p, encoding='utf-8').read()


def rep(old, new, n=1):
    global s
    c = s.count(old)
    assert c == n, (c, old[:110])
    s = s.replace(old, new)


SHARE_OLD = ('which accounts for 70--85\\% of the total at the transformation maximum, with limited sea ice extent restricting '
             'the haline contribution from brine rejection to a secondary role (15--30\\%; green line).')
SHARE_NEW = ('which accounts for 63--90\\% of the summed magnitude of the three terms at the transformation maximum, with limited '
             'sea ice extent restricting the haline terms, sea ice and other freshwater fluxes together, to a secondary role '
             '(10--37\\%; green and blue lines).')
DOUBLE_OLD = ('moves the partition in the direction they report. The sea ice contribution at the transformation maximum roughly '
              'doubles, from 0.86 to 1.84~Sv (Fig. \\ref{domain}).')
DOUBLE_NEW = ('moves the partition in the direction they report, but only slightly at the transformation maximum. There the '
              'thermal share of PI falls from 79\\% to 76\\%, and the sea ice term stays near 2.6~Sv (Fig. \\ref{domain}).')
SAM_OLD = ('However, the magnitude and mechanism of this SAM-WMT coupling differ substantially between interglacial and glacial '
           'periods. For PI, the maximum impacts occur at $\\sigma_2 = 36.7$ kg/m$^3$ with the SAM-induced WMT anomaly being 3.2 Sv,  '
           'while in the mid-Holocene and the Last Interglacial, the maximum WMT anomaly between high and low SAM years can reach '
           'about 8 Sv, occurring at $\\sigma_2 = 36.5$ kg/m$^3$, representing the strongest SAM-WMT coupling among all climate '
           'states. During glacial periods, peak anomalies are more modest (+4 Sv for LGM at $\\sigma_2 = 37.5$~kg/m$^3$; +4.2 Sv '
           'for MIS3 at $\\sigma_2 = 37.1$~kg/m$^3$) and shifted  toward higher density classes, reflecting colder and saltier '
           'background conditions.')
SAM_NEW = ('The size of this SAM-WMT coupling is similar in all climate states, while its mechanism differs between interglacial '
           'and glacial periods. The largest anomaly between high and low SAM years is 6.1~Sv in PI (at $\\sigma_2 = 36.1$--$36.2$~kg~m$^{-3}$), '
           '8.2~Sv in the mid-Holocene (36.6--36.7~kg~m$^{-3}$) and 10.4~Sv in the Last Interglacial (36.5--36.6~kg~m$^{-3}$). '
           'In the glacial states it reaches 10.2~Sv in LGM (37.4--37.5~kg~m$^{-3}$) and 7.0~Sv in MIS3 (37.1--37.2~kg~m$^{-3}$), '
           'shifted toward higher density classes, which reflects the colder and saltier background conditions.')

if mode == 'revised':
    # remove the notes added by edits_1127g
    n0 = s.count('% claude（11-27，0.1 格）：')
    s = re.sub(r'\n% claude（11-27，0\.1 格）：[^\n]*', '', s)
    assert n0 == 3 and '% claude（11-27，0.1 格）' not in s, n0
    rep('modulates transformation by 3--8 Sv between its positive and negative phases',
        'modulates transformation by 6--10 Sv between its positive and negative phases')
    rep(SHARE_OLD, SHARE_NEW)
    rep('with the sea ice term contributing 0.79~Sv against 0.06~Sv from heat in the annual mean',
        'with the sea ice term contributing on average 0.44~Sv against 0.04~Sv from heat in the annual mean')
    rep(SAM_OLD, SAM_NEW)
    rep(DOUBLE_OLD, DOUBLE_NEW)
    rep('the PI transformation is haline dominated, with 0.79~Sv from sea ice against 0.06~Sv from heat.',
        'the PI transformation is haline dominated, with on average 0.44~Sv from sea ice against 0.04~Sv from heat.')
    rep(r'and the thermal share of the transformation falls to 13--18\% while the sea ice term becomes dominant.',
        r'and the thermal share of the transformation falls to 17--19\% while the sea ice term becomes dominant.')
    rep(r'There the glacial states are 15\% and 18\% thermally driven and the interglacial states 53--78\% (Fig. \ref{domain})',
        r'There the glacial states are 16\% and 19\% thermally driven and the interglacial states 54--83\% (Fig. \ref{domain})')
    rep('with composite anomalies of 3--8 Sv between high and low phases',
        'with composite anomalies of 6--10 Sv between high and low phases')
    rep('These auxiliary calculations use the climatological monthly means and are otherwise identical to the main calculation.',
        'These auxiliary calculations use the same 100 years of monthly output and are otherwise identical to the main calculation.')
elif mode == 'letter':
    rep('modulates transformation by 3--8 Sv between its positive and negative phases',
        'modulates transformation by 6--10 Sv between its positive and negative phases')
    rep('which accounts for 70-85\\% of the total at the transformation maximum, with limited sea ice extent restricting the '
        'haline contribution from brine rejection to a secondary role (15-30\\%; green line).',
        'which accounts for 63-90\\% of the summed magnitude of the three terms at the transformation maximum, with limited sea '
        'ice extent restricting the haline terms, sea ice and other freshwater fluxes together, to a secondary role '
        '(10-37\\%; green and blue lines).')
    rep('The thermal term accounts for 70–85\\% of the total at the transformation maximum in the interglacials, against 15–30\\% '
        'for the haline term, and falls to 12–17\\% in the glacial states.',
        'The thermal term accounts for 63–90\\% of the summed magnitude of the three terms at the transformation maximum in the '
        'interglacials, against 10–37\\% for the haline terms, and falls to 17–19\\% in the glacial states.')
    rep(DOUBLE_OLD.replace('Fig. \\ref{domain}', 'Fig. 8'), DOUBLE_NEW.replace('Fig. \\ref{domain}', 'Fig. 8'))
    rep('the PI transformation is haline dominated, with 0.79~Sv from sea ice against 0.06~Sv from heat.',
        'the PI transformation is haline dominated, with on average 0.44~Sv from sea ice against 0.04~Sv from heat.')
    rep(r'and the thermal share of the transformation falls to 13-18\% while the sea ice term becomes dominant.',
        r'and the thermal share of the transformation falls to 17-19\% while the sea ice term becomes dominant.')
    rep(r'the glacial states are 15\% (LGM) and 18\% (MIS3) thermally driven at the transformation maximum, against 53--78\% in the interglacial states',
        r'the glacial states are 16\% (LGM) and 19\% (MIS3) thermally driven at the transformation maximum, against 54--83\% in the interglacial states', n=2)
    rep(r'There the glacial states are 15\% and 18\% thermally driven and the interglacial states 53--78\% (Fig. 8)',
        r'There the glacial states are 16\% and 19\% thermally driven and the interglacial states 54--83\% (Fig. 8)')
else:
    sys.exit('mode must be revised or letter')

assert '\t' not in s and '\r' not in s
open(p, 'w', encoding='utf-8').write(s)
print('ok', mode, p)
