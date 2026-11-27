#!/usr/bin/env python3
r"""11-27 14:31 (user GQ + questions-ef): put back the wording of the TexPage submission (A) that the revision lost
because it was built from the 05-18 draft (B). Hunks from submitted_vs_draft.md / provenance.md.
Restored where the revision still had the draft text: Introduction paragraphs 1 and 2 (WMT) whole, paragraph 4
(knorr2021 sentence, single Skinner2017 citation), 'precondition', Fig. 2 panel pointers a-c, d,e, f-t,
'densification.', 'contributors, i.e.,', Discussion knorr2021 clause and the commented-out 'Moreover' passage,
ventilation-age paragraph order and wording outside the reviewer-driven sentences, future-implications paragraph
and its position before the single-model paragraph, GHG citation list without buiron2011, 'transformation framework',
funding text, Fig. 2 and A11 line legend, Fig. 9 caption without numbers and without the 270 degree rotation.
Typos of A corrected on the way: though which -> through which, Southern ocean -> Southern Ocean, firstly -> first,
remove -> removes, These work point -> These studies point.
Kept as revised (reasons in consistency_table.md): reviewer-driven rewrites (Section 2.3 units and numbers, SAM
caption thresholds, A2 figure, data availability, radiocarbon sentence, age caveats, glacial WMT numbers), the M(rho)
definition (A's 'positive F indicates accumulation' is wrong), '$-$150', the A3 pointer (open item 8).
Usage: edits_1127j_restore.py revised <revised.tex> <A.tex> <B.tex> | bib <ref.bib> <A Ref.bib>"""
import re
import sys

mode = sys.argv[1]


def rep(s, old, new, n=1):
    c = s.count(old)
    assert c == n, (c, old[:120])
    return s.replace(old, new)


if mode == 'revised':
    p, pa, pb = sys.argv[2:5]
    s = open(p, encoding='utf-8').read()
    A = open(pa, encoding='utf-8').read().splitlines()
    B = open(pb, encoding='utf-8').read().splitlines()
    # Introduction paragraph 1 (A L59 / B L60) and the WMT paragraph(s) (A L72 / B L73-76), whole
    a59 = A[58]
    assert a59.startswith('The Southern Ocean plays an important role')
    a59 = rep(a59, 'though which the Southern ocean regulates', 'through which the Southern Ocean regulates')
    s = rep(s, B[59] + '\n', a59 + '\n')
    a72 = A[71]
    assert a72.startswith('The water mass transformation (WMT) framework')
    a72 = rep(a72, 'This method was firstly introduced to relate', 'This method was first introduced to relate')
    a72 = rep(a72, 'a freshwater pump that remove freshwater', 'a freshwater pump that removes freshwater')
    s = rep(s, '\n'.join(B[72:76]) + '\n', a72 + '\n')
    # Introduction paragraph 4
    s = rep(s, r'during glacial periods \cite{Skinner2017}, a slowdown', r'during glacial periods, a slowdown')
    s = rep(s, r'\cite{Marzocchi2017}. This points to a thermal-to-haline shift',
            r'\cite{Marzocchi2017}. More recent modeling studies further emphasize the importance of Southern Ocean '
            r'salinity structure and sea ice for glacial overturning stability and deep-ocean isolation '
            r'\cite{knorr2021salty}. These studies point to a possible thermal-to-haline shift')
    # Results
    s = rep(s, 'the surface preconditioning for dense water formation', 'the surface precondition for dense water formation')
    s = rep(s, r'across all simulated interglacial states (Fig. \ref{wmt_jja}). This',
            r'across all simulated interglacial states (Fig. \ref{wmt_jja}a-c). This')
    s = rep(s, r'($\sigma_2 = 37.1$-$37.6$ kg m$^{-3}$; Fig. \ref{wmt_jja}).',
            r'($\sigma_2 = 37.1$-$37.6$ kg m$^{-3}$; Fig. \ref{wmt_jja}d,e).')
    s = rep(s, r'intensifying haline-driven densification at higher density classes ($\sigma_2 > 37$ kg m$^{-3}$).',
            r'intensifying haline-driven densification.')
    s = rep(s, r"including the Ross Sea, Weddell Sea, and Ad\'{e}lie regions.",
            r"including the Ross Sea, Weddell Sea, and Ad\'{e}lie regions (Fig. \ref{wmt_jja}f-t).")
    s = rep(s, 'through two distinct contributors: thermal forcing', 'through two distinct contributors, i.e., thermal forcing')
    # Discussion
    s = rep(s, r'aligns with the sea ice pump hypothesis \cite{Abernathey2016}. In this regime',
            r'aligns with the sea ice pump hypothesis \cite{Abernathey2016} and with recent studies emphasizing the '
            r'importance of Southern Ocean salinity stratification and sea ice in regulating glacial overturning '
            r'circulation and abyssal isolation \cite{knorr2021salty}. In this regime')
    s = rep(s, ' Moreover, our paleoclimate analysis suggests that projected sea ice decline',
            ' %Moreover, our paleoclimate analysis suggests that projected sea ice decline')
    s = rep(s, 'Finally, we evaluate the simulated deep ocean ventilation ages. Benthic-planktonic',
            'Moreover, we evaluate the simulated deep ocean ventilation ages. Our glacial simulations present '
            'substantially increased ventilation ages, with Pacific ages reaching 1500 years compared to less than '
            '500 years during interglacials. Benthic-planktonic')
    s = rep(s, r'Our  simulated LGM ideal age increase of about 1000 years at 4000 m depth is of similar magnitude to '
               r'radiocarbon-derived estimates of  689$\pm$53~$^{14}$C-yr \cite{Skinner2017}, though our model somewhat '
               r'overestimates the age increase. However, direct comparison',
            r'Our  simulated LGM ideal age increase in deep ocean is of similar magnitude to radiocarbon-derived '
            r'estimates of  689$\pm$53~$^{14}$C-yr \cite{Skinner2017}. However, it should be noted that direct comparison')
    s = rep(s, r' Our glacial simulations present substantially increased ventilation ages, with Pacific ages reaching '
               r'1500 years compared to less than 500 years during interglacials, consistent with radiocarbon proxy '
               r'reconstructions \cite{Skinner2017}.', '')
    single = ('While this study is based on a single model, the mechanisms identified here provide a framework for '
              'systematic inter-model comparison. Applying this analysis across the CMIP/PMIP ensemble would help address '
              'the current large inter-model spread in glacial-interglacial WMT and better constrain its implications '
              'for deep water formation.')
    i = s.index(single)
    j = s.index('These paleoclimate results carry direct implications for the future Southern Ocean.')
    assert s[i + len(single):j] == '\n\n', repr(s[i + len(single):j])
    k = s.index('\n', j)
    impl = s[j:k]
    impl = rep(impl, 'These paleoclimate results carry direct implications', 'Our results carry direct implications')
    impl = rep(impl, r'favouring a more thermally driven regime and, at the extreme, the cessation of open-ocean deep '
                     r'convection \cite{deLavergne2014}.', 'favoring a more thermally driven regime.')
    impl = rep(impl, 'and our analysis shows that this balance is itself climate-state dependent. This implies that '
                     'projections of Southern Ocean ventilation and its associated carbon uptake will remain uncertain '
                     'until the thermal and haline components of surface buoyancy forcing are evaluated separately, '
                     'alongside bulk overturning metrics.',
               'and our analysis indicates that this balance is potentially climate-state dependent.')
    s = s[:i] + impl + '\n\n' + single + s[k:]
    # Methods
    s = rep(s, r'schilt2010glacial,buiron2011taldice,schneider2013reconstruction', r'schilt2010glacial,schneider2013reconstruction')
    s = rep(s, 'using the water mass transformation (WMT) framework', 'using the water mass transformation framework')
    s = rep(s, '%% TODO before submission: acknowledgements and funding text.\n', '')
    s = rep(s, 'This research was supported by... We acknowledge helpful discussions with...',
            'This work is supported by the National Key Research and Development Program of China (Grant No. '
            '2024YFF0506604), the Southern Marine Science and Engineering Guangdong Laboratory (Zhuhai) (grant no. '
            'SML2023SP204), and the Ocean Negative Carbon Emissions (ONCE) Program.')
    # captions
    s = rep(s, r'Lines show the mean values; shading denotes',
            r'Lines denote total WMT (black), heat-induced WMT (red), sea ice-induced WMT (green), and WMT driven by '
            r'other freshwater fluxes (net precipitation plus river runoff) (blue). Shading denotes', n=2)
    s = rep(s, r'\centering\includegraphics[angle=270,width=\textwidth]{figures/frame.pdf}',
            r'\centering\includegraphics[width=\textwidth]{figures/frame.pdf}')
    s = rep(s, r'surface cooling and dense water formation at $\sigma_2 \approx 36.4$--$36.8$~kg~m$^{-3}$ (70--81~Sv).',
            r'surface cooling and dense water formation.')
    s = rep(s, r'producing denser waters ($\sigma_2 > 37.0$~kg~m$^{-3}$) at lower rates (50--59~Sv).',
            r'producing denser waters at lower rates.')
    s = rep(s, r'Ventilation ages at 4000~m depth correspondingly', r'Ventilation ages at 4000 m depth correspondingly')
    for k_ in ('sime2013southern', 'chadwick2022compilation', 'volker2013responses', 'mandal2021roles', 'knorr2021salty'):
        assert re.search(r'\\cite\{[^}]*\b%s\b' % k_, s), k_
    assert 'deLavergne2014' not in s and 'buiron2011taldice' not in s
    open(p, 'w', encoding='utf-8').write(s)
    print('ok revised', p)
elif mode == 'bib':
    p, pa = sys.argv[2:4]
    s = open(p, encoding='utf-8').read()
    a = open(pa, encoding='utf-8').read()
    add = []
    for k_ in ('sime2013southern', 'chadwick2022compilation', 'volker2013responses', 'mandal2021roles', 'knorr2021salty'):
        assert not re.search(r'@\w+\{%s,' % k_, s), k_ + ' already there'
        m = re.search(r'@\w+\{%s,.*?\n\}' % k_, a, re.S)
        assert m, k_
        add.append(m.group(0))
    s = s.rstrip('\n') + '\n\n' + '\n\n'.join(add) + '\n'
    open(p, 'w', encoding='utf-8').write(s)
    print('ok bib', len(add), 'entries appended to', p)
else:
    sys.exit('mode must be revised or bib')
