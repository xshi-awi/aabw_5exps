#!/usr/bin/env python3
"""11-27 12:04 user button GN: WMT in sigma2 bins 0.1 kg m-3 wide everywhere ("换0.1全部重算").
Plain number changes only. Sentences whose conclusion changes, or whose old number has no reproducible
source, are left as they are and get a % note (listed for the user in pass_20261127c/bin0p1_table.md).
Usage: edits_1127g_bin0p1.py revised <revised.tex>   |   edits_1127g_bin0p1.py letter <response_new.tex>"""
import sys

mode, p = sys.argv[1], sys.argv[2]
s = open(p, encoding='utf-8').read()


def rep(old, new, n=1):
    global s
    c = s.count(old)
    assert c == n, (c, old[:100])
    s = s.replace(old, new)


NOTE = '% claude（11-27，0.1 格）：'
if mode == 'revised':
    # Results 2.2
    rep(r'exhibits a WMT maximum of 60--80~Sv at $\sigma_2 \approx 36.5$-$36.8$ kg m$^{-3}$ across all simulated interglacial states',
        r'exhibits a WMT maximum of 70--81~Sv at $\sigma_2 \approx 36.4$-$36.8$ kg m$^{-3}$ across all simulated interglacial states')
    rep(r'The winter WMT maximum weakens substantially to 48-58 Sv while shifting to higher density classes ($\sigma_2 = 37.0$-$37.5$ kg m$^{-3}$; Fig. \ref{wmt_jja}).',
        r'The winter WMT maximum weakens substantially to 50-59 Sv while shifting to higher density classes ($\sigma_2 = 37.1$-$37.6$ kg m$^{-3}$; Fig. \ref{wmt_jja}).')
    rep('''The thermal dominance reported here is therefore a property of the integral over all density classes, and we quantify its sensitivity to the choice of integration domain and density range in the Discussion.
''', '''The thermal dominance reported here is therefore a property of the integral over all density classes, and we quantify its sensitivity to the choice of integration domain and density range in the Discussion.
''' + NOTE + '''本段 70--85\\%、15--30\\% 和 0.79/0.06 Sv 没改，待你定（见 pass_20261127c/bin0p1_table.md）。
''')
    # Discussion: domain paragraph left unchanged (conclusion changes at 0.1), note only
    rep('''The sea ice contribution at the transformation maximum roughly doubles, from 0.86 to 1.84~Sv (Fig. \\ref{domain}).''',
        '''The sea ice contribution at the transformation maximum roughly doubles, from 0.86 to 1.84~Sv (Fig. \\ref{domain}).
''' + NOTE + '''0.1 格下 Fig. 8 的峰值处海冰项是 2.25（60°S 以南）对 2.17 Sv（海冰区），不再翻倍，这句待你定。
''')
    # Discussion: formation location (A13)
    rep(r'forms mainly over ice-free deep water, which carries 69\% of it, while the continental shelf and slope, where the bottom is shallower than 1000~m, carry 20--24\%.',
        r'forms mainly over ice-free deep water, which carries 68--71\% of it, while the continental shelf and slope, where the bottom is shallower than 1000~m, carry 16--25\%.')
    rep(r'In the glacial states the class of the transformation maximum forms under sea ice offshore (89--91\%), and the densest classes',
        r'In the glacial states the class of the transformation maximum forms under sea ice offshore (87--90\%), and the densest classes')
    # Discussion: summary paragraph
    rep(r'with transformation maxima of 60-80~Sv at density classes of $\sigma_2 = 36.5$-$36.8$ kg m$^{-3}$',
        r'with transformation maxima of 70-81~Sv at density classes of $\sigma_2 = 36.4$-$36.8$ kg m$^{-3}$')
    rep(r'The winter transformation maximum weakens to below 60~Sv and moves to $\sigma_2 = 37.0$--$37.5$~kg~m$^{-3}$, and the thermal share of the transformation falls to 12--17\% while the sea ice term becomes dominant.',
        r'The winter transformation maximum weakens to below 60~Sv and moves to $\sigma_2 = 37.1$--$37.6$~kg~m$^{-3}$, and the thermal share of the transformation falls to 13--18\% while the sea ice term becomes dominant.')
    # SAM paragraph and abstract: left unchanged, note only
    rep('''During glacial periods, peak anomalies are more modest (+4 Sv for LGM at $\\sigma_2 = 37.5$~kg/m$^3$; +4.2 Sv for MIS3 at $\\sigma_2 = 37.1$~kg/m$^3$) and shifted  toward higher density classes, reflecting colder and saltier background conditions.''',
        '''During glacial periods, peak anomalies are more modest (+4 Sv for LGM at $\\sigma_2 = 37.5$~kg/m$^3$; +4.2 Sv for MIS3 at $\\sigma_2 = 37.1$~kg/m$^3$) and shifted  toward higher density classes, reflecting colder and saltier background conditions.
''' + NOTE + '''本段 SAM 数字（3.2、8、4、4.2 Sv）连 0.2 格都对不上；0.1 格最大异常 PI 6.1、MH 8.2、LIG 10.4、LGM 10.2、MIS3 7.0 Sv，冰期不再「更小」，待你定。摘要和讨论的 3--8 Sv 同。
''')
    # Discussion: sea ice zone shares (Fig. 8, 0.1 bins: LGM 15%, MIS3 18%, interglacial 53/78/77%)
    rep(r'There the glacial states are 15\% and 17\% thermally driven and the interglacial states 53--70\% (Fig. \ref{domain})',
        r'There the glacial states are 15\% and 18\% thermally driven and the interglacial states 53--78\% (Fig. \ref{domain})')
    # Methods
    rep(r'we bin the surface fluxes into density classes with a bin width ($\Delta \rho$) of 0.2 kg m$^{-3}$.',
        r'we bin the surface fluxes into density classes with a bin width ($\Delta \rho$) of 0.1 kg m$^{-3}$.')
    # captions and widths
    rep(r'\caption{JJA-mean water mass transformation rates [Sv] as a function of density $\sigma_2$ [kg m$^{-3}$] for (a-e) Southern Ocean,',
        r'\caption{JJA-mean water mass transformation rates [Sv] as a function of density $\sigma_2$ [kg m$^{-3}$], in density bins 0.1 kg m$^{-3}$ wide, for (a-e) Southern Ocean,')
    rep(r'and the integrated area of each domain, are given in each panel.',
        r'and the integrated area of each domain, are given below each panel.')
    rep(r'\includegraphics[width=0.9\textwidth]{figures/plot_wmt_4regions_5exps_winter_from_100years.pdf}',
        r'\includegraphics[width=\textwidth]{figures/plot_wmt_4regions_5exps_winter_from_100years.pdf}')
    rep(r'\includegraphics[width=0.9\textwidth]{figures/wmt_sam_composite_4regions_5exps.pdf}',
        r'\includegraphics[width=\textwidth]{figures/wmt_sam_composite_4regions_5exps.pdf}')
    rep(r'\centering\includegraphics[width=0.8\textwidth]{figures/plot_wmt_4regions_5exps_annual_from_100years.pdf}',
        r'\centering\includegraphics[width=\textwidth]{figures/plot_wmt_4regions_5exps_annual_from_100years.pdf}')
elif mode == 'letter':
    # quotes of the manuscript (follow the edits above)
    rep(r'forms mainly over ice-free deep water, which carries 69\% of it, while the continental shelf and slope, where the bottom is shallower than 1000~m, carry 20--24\%.',
        r'forms mainly over ice-free deep water, which carries 68--71\% of it, while the continental shelf and slope, where the bottom is shallower than 1000~m, carry 16--25\%.')
    rep(r'In the glacial states the class of the transformation maximum forms under sea ice offshore (89--91\%), and the densest classes',
        r'In the glacial states the class of the transformation maximum forms under sea ice offshore (87--90\%), and the densest classes')
    rep(r'The winter transformation maximum weakens to below 60 Sv and moves to σ\textsubscript{2} = 37.0-37.5 kg m\textsuperscript{-3}, and the thermal share of the transformation falls to 12-17\% while the sea ice term becomes dominant.',
        r'The winter transformation maximum weakens to below 60 Sv and moves to σ\textsubscript{2} = 37.1-37.6 kg m\textsuperscript{-3}, and the thermal share of the transformation falls to 13-18\% while the sea ice term becomes dominant.')
    # replies
    rep(r'forms mainly by heat loss over ice-free deep water (69\%). The densest classes',
        r'forms mainly by heat loss over ice-free deep water (68--71\%). The densest classes')
    rep(r'the glacial states are 15\% (LGM) and 17\% (MIS3) thermally driven at the transformation maximum, against 53--70\% in the interglacial states',
        r'the glacial states are 15\% (LGM) and 18\% (MIS3) thermally driven at the transformation maximum, against 53--78\% in the interglacial states', n=2)
    rep(r'There the glacial states are 15\% and 17\% thermally driven and the interglacial states 53--70\% (Fig. 8)',
        r'There the glacial states are 15\% and 18\% thermally driven and the interglacial states 53--78\% (Fig. 8)')
    rep(r'the class of the transformation maximum, the 0.2 kg m\textsuperscript{-3} bin of manuscript Fig. 2 in which the curve peaks.',
        r'the class of the transformation maximum, the 0.1 kg m\textsuperscript{-3} bin of manuscript Fig. 2 in which the curve peaks.')
    rep(r'(σ\textsubscript{2} = 36.4--36.8 kg m\textsuperscript{-3}, 69--81 Sv) forms mainly by heat loss over ice-free deep water (69\%), and the continental shelf and slope, where the bottom is shallower than 1000 m, contribute 20--24\%.',
        r'(σ\textsubscript{2} = 36.4--36.8 kg m\textsuperscript{-3}, 70--81 Sv) forms mainly by heat loss over ice-free deep water (68--71\%), and the continental shelf and slope, where the bottom is shallower than 1000 m, contribute 16--25\%.')
    rep(r'The class of the transformation maximum forms under the ice offshore (89--91\%), mainly by brine rejection,',
        r'The class of the transformation maximum forms under the ice offshore (87--90\%), mainly by brine rejection,')
    rep(r'This densest class lies on the steep dense side of the transformation maximum and is narrower than the 0.2 kg m\textsuperscript{-3} density bins of manuscript Fig. 2, so in that figure it is merged with the thermally driven maximum.',
        r'This densest class lies on the steep dense side of the transformation maximum. In manuscript Fig. 2, now drawn with density bins 0.1 kg m\textsuperscript{-3} wide, it appears as the small tail beyond the sharp drop near σ\textsubscript{2} = 36.9 kg m\textsuperscript{-3}, where the sea ice term carries almost all of the remaining transformation.')
    rep(r'(69\% in the interglacial states)', r'(68--71\% in the interglacial states)')
else:
    sys.exit('mode must be revised or letter')

assert '\t' not in s and '\r' not in s
open(p, 'w', encoding='utf-8').write(s)
print('ok', mode, p)
