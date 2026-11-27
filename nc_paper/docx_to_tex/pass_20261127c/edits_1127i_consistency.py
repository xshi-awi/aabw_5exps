#!/usr/bin/env python3
r"""11-27 13:38 final check of letter against revision (user via questions-ef): pure text mismatches only.
Content choices are listed in consistency_table.md, not made here.
revised: comma after Here (R1 #8); density pointers Fig. 1i-l / 1j -> A1c,d / A1b (Fig. 1i-l is sea ice);
'while The' -> 'while the'; dashed lines are the 15 W m-2 contour of A7 (caption), so the pointer moves next to
the region; LIG brine response 41 mSv (sam_push_pull.json 41.47); citations with author names (R1 #31, Berger);
'Each paleoclimate experiment' (PI ran 1,500 years, Methods); component full names (R1 #30 promised);
Fig. 9 caption numbers at 0.1 bins (as Section 2.2); A12 cited before A13 (R1 #7 supplementary order).
letter: Heuze 28 of 35 (as the manuscript); stale 0.2-bin numbers in two replies; 1.9 deg is one T63 grid row
(47.564 -> 49.429 S); 'next paragraph'; 'surface node'; the quotes follow the manuscript edits above.
Usage: edits_1127i_consistency.py revised <revised.tex> | letter <response_new.tex>"""
import sys

mode, p = sys.argv[1], sys.argv[2]
s = open(p, encoding='utf-8').read()


def rep(old, new, n=1):
    global s
    c = s.count(old)
    assert c == n, (c, old[:110])
    s = s.replace(old, new)


if mode == 'revised':
    rep(r'Here the enhanced glacial salinification', r'Here, the enhanced glacial salinification')
    rep(r'In the glacial states the surface density anomalies (Fig. \ref{sst}i-l) follow the salinity field.',
        r'In the glacial states the surface density anomalies (Fig. \ref{wind}c,d) follow the salinity field.')
    rep(r'is close to density compensated (Fig. \ref{sst}j)', r'is close to density compensated (Fig. \ref{wind}b)')
    rep('while\nThe Last Interglacial exhibits pronounced', 'while\nthe Last Interglacial exhibits pronounced')
    rep(r'between 50 and 70°S, north of the winter ice edge (dashed lines in Fig.~\ref{sam_heat}a-c).',
        r'between 50 and 70°S (dashed lines in Fig.~\ref{sam_heat}a-c), north of the winter ice edge.')
    rep(r'by 51, 12 and 42~mSv', r'by 51, 12 and 41~mSv')
    rep('mixing scheme following\n  \\cite{timmermann2004parameterization}',
        'mixing scheme following Timmermann and Beckmann\n  \\cite{timmermann2004parameterization}')
    rep(r'Orbital parameters are calculated following \cite{berger1977long}',
        r'Orbital parameters are calculated following Berger \cite{berger1977long}')
    rep(r'Each experiment was integrated for 1000 years', r'Each paleoclimate experiment was integrated for 1000 years')
    rep(r'We employ the AWI-ESM2, a state-of-the-art', r'We employ the AWI-ESM2 (Alfred Wegener Institute Earth System Model, version 2), a state-of-the-art')
    rep(r'The model comprises the ECHAM6 atmospheric component', r'The model comprises the ECHAM6 (European Centre Hamburg model, version 6) atmospheric component')
    rep(r'coupled to the JSBACH land-surface model', r'coupled to the JSBACH (Jena Scheme for Biosphere-Atmosphere Coupling in Hamburg) land-surface model')
    rep(r'The ocean-sea ice component is FESOM2, which', r'The ocean-sea ice component is FESOM2 (Finite-volumE Sea ice-Ocean Model, version 2), which')
    rep(r'dense water formation at $\sigma_2 \approx 36.5$--$36.8$~kg~m$^{-3}$ (60--80~Sv).',
        r'dense water formation at $\sigma_2 \approx 36.4$--$36.8$~kg~m$^{-3}$ (70--81~Sv).')
    rep(r'at lower rates (48--58~Sv).', r'at lower rates (50--59~Sv).')
    rep(r'by open-ocean convection in the interior of the Weddell gyre near 68°S.',
        r'by open-ocean convection in the interior of the Weddell gyre near 68°S (Supplementary Fig. \ref{mld_obs}).')
    # \citet prints '(author?) [n]' with the numeric sn-nature style: write the names (authors from ref.bib)
    for key, names, n in [('sidorenko2019evaluation', 'Sidorenko et al.', 1), ('Shi2025GRL', 'Shi et al.', 3),
                          ('Shi2022JCLI', 'Shi et al.', 2), ('Shi2023CP', 'Shi et al.', 1),
                          ('Bartlein2011', 'Bartlein et al.', 1), ('TurneyJones2010', 'Turney and Jones', 1)]:
        rep(r'\citet{%s}' % key, r'%s \cite{%s}' % (names, key), n=n)
    assert r'\citet{' not in s
elif mode == 'letter':
    rep('Heuzé (2021) found that almost all CMIP6 models form', 'Heuzé (2021) found that 28 of 35 CMIP6 models form', n=2)
    rep(r'moves to σ\textsubscript{2} = 37.0–37.5 kg m\textsuperscript{-3}, with the thermal share falling to 12–17\%.',
        r'moves to σ\textsubscript{2} = 37.1–37.6 kg m\textsuperscript{-3}, with the thermal share falling to 17–19\%.')
    rep(r'transformation maximum lies at σ\textsubscript{2} ≈ 36.5--36.8 kg m\textsuperscript{-3}',
        r'transformation maximum lies at σ\textsubscript{2} ≈ 36.4--36.8 kg m\textsuperscript{-3}')
    rep(r'in LGM, 1.9° further south and about 13\% stronger', r'in LGM, one grid row (1.9°) further south and about 13\% stronger')
    rep(r'it links Fig. 1i-l to the mixed layer changes described in the same paragraph.',
        r'it links Fig. 1i-l to the mixed layer changes described in the next paragraph.')
    rep(r'computed thermodynamically at each mesh and enter the ocean', r'computed thermodynamically at each surface node and enter the ocean')
    rep(r'while The Last Interglacial exhibits pronounced', r'while the Last Interglacial exhibits pronounced')
    rep(r'between 50 and 70°S, north of the winter ice edge (dashed lines in Fig. A7a-c).',
        r'between 50 and 70°S (dashed lines in Fig. A7a-c), north of the winter ice edge.')
    rep(r'by 51, 12 and 42 mSv', r'by 51, 12 and 41 mSv')
    rep(r'by 51, 12 and 42~mSv', r'by 51, 12 and 41~mSv')
    rep(r'Orbital parameters are calculated following \textit{(Berger, 1977)}', r'Orbital parameters are calculated following Berger (1977)')
    rep(r'by open-ocean convection in the interior of the Weddell gyre near 68°S.',
        r'by open-ocean convection in the interior of the Weddell gyre near 68°S (Supplementary Fig. A12).')
else:
    sys.exit('mode must be revised or letter')

assert '\t' not in s and '\r' not in s
open(p, 'w', encoding='utf-8').write(s)
print('ok', mode, p)
