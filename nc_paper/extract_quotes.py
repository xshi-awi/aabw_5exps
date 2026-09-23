#!/usr/bin/env python
"""
Pull verbatim passages out of build/revised.tex for quotation in the response
letter, and convert them to the plain prose the letter uses.

The point is that every italic quotation in the letter must be the real sentence
from the manuscript, not a paraphrase. A reviewer who checks will otherwise find
a mismatch.
"""
import re
import json
from pathlib import Path

SRC = Path('build/revised.tex')
OUT = Path('quotes.json')
s = SRC.read_text()


def clean(t):
    """LaTeX -> plain prose suitable for the markdown letter."""
    # \citet puts the author's name in the sentence, so dropping it leaves a
    # verbless clause ("find precipitation in reasonable agreement with..."),
    # which reads as a mistake in the letter. Substitute the name instead.
    CITET = {'Bartlein2011': 'Bartlein et al.',
             'TurneyJones2010': 'Turney and Jones',
             'Shi2022JCLI': 'Shi et al. (2022)',
             'Shi2023CP': 'Shi et al. (2023)',
             'Shi2025GRL': 'Shi et al. (2025)',
             'sidorenko2019evaluation': 'Sidorenko et al. (2019)'}
    t = re.sub(r'\\citet\{([^}]*)\}',
               lambda m: CITET.get(m.group(1), m.group(1)), t)
    t = re.sub(r'\\cite[a-zA-Z]*\{[^}]*\}', '', t)
    # map labels to the figure numbers a reviewer will see, rather than dropping them
    REF = {'sst': '1', 'mld': 'A2', 'wind': 'A3', 'wmt_jja': '2',
           'density_heat': '3', 'density_fwf': '4', 'sam_wmt': 'S4',
           'sam_heat': 'S2', 'sam_fwf': 'S3', 'age': '5', 'moc': '6',
           'age_global': '7', 'domain': '8', 'reso': 'A1',
           's_common_axis': 'S5', 'sam_pushpull': 'S1', 'sam_windice': 'S6',
           's2': 'A4', 'schematic': '9', 'tab:bc': '1'}
    t = re.sub(r'\\ref\{([^}]*)\}', lambda m: REF.get(m.group(1), '?'), t)
    t = re.sub(r'\\label\{[^}]*\}', '', t)
    t = t.replace('\\%', '%').replace('\\&', '&').replace('~', ' ')
    t = t.replace('$^{\\circ}$', '°').replace('$^\\circ$', '°')
    # Emit the Unicode symbols the letter already uses, rather than an ASCII
    # transliteration. \sigma_2 occurs inside larger maths such as
    # $\sigma_2 = 37.0$, so substitute the command itself; the generic
    # \\[a-zA-Z]+ strip below would otherwise leave a bare "_2", and the
    # surrounding $ would survive as a literal dollar sign.
    t = t.replace('\\sigma_2', '\u03c3\u2082').replace('\\gamma_n', '\u03b3\u2099')
    t = t.replace('\\sigma', '\u03c3').replace('\\gamma', '\u03b3')
    t = t.replace('\\ge', '\u2265').replace('\\le', '\u2264')
    t = t.replace('\\approx', '\u2248').replace('\\sim', '~')
    t = t.replace('$\\alpha$', 'alpha').replace('$\\beta$', 'beta')
    t = t.replace('$\\pm$', '\u00b1').replace('$\\times$', '\u00d7')
    t = t.replace('\\,', ' ').replace('--', '-')
    t = re.sub(r'\$([^$]*)\$', r'\1', t)
    t = t.replace('\\ldots', '...').replace("\\'e", 'e').replace('\\v{c}', 'c')
    t = re.sub(r'\\[a-zA-Z]+', '', t)
    t = t.replace('{', '').replace('}', '')
    t = t.replace('^{-3}', '⁻³').replace('^{-1}', '⁻¹').replace('^{-2}', '⁻²')
    t = t.replace('^-3', '⁻³').replace('^-1', '⁻¹').replace('^-2', '⁻²')
    t = re.sub(r'\s+', ' ', t)
    return t.strip()


def grab(start, end=None, n=1):
    """Return the text beginning at `start`; to `end` if given, else n sentences."""
    i = s.find(start)
    if i < 0:
        return None
    if end:
        j = s.find(end, i)
        if j < 0:
            return None
        return clean(s[i:j + len(end)])
    # n sentences
    seg = s[i:i + 4000]
    out, cnt = [], 0
    for m in re.finditer(r'[^.]*\.', seg):
        out.append(m.group(0))
        cnt += 1
        if cnt >= n:
            break
    return clean(''.join(out))


Q = {}

# ---- abstract, the SAM bridge sentence (R1 comments 2, 3, 4)
Q['abstract_sam'] = grab(
    'Because the balance between these two pathways',
    'when it is ice covered.')

Q['abstract_vent'] = grab(
    'Weaker glacial transformation coincides',
    'reaching 1500 years in the Pacific.')

# ---- introduction, expanded SAM (R1 comment 5)
Q['intro_sam'] = grab(
    'The Southern Annular Mode is the leading mode',
    'is not known.')

# ---- Chen et al. in the introduction (R2 major 2)
Q['intro_chen'] = grab(
    'A recent study using the same ocean model',
    'across several climate states.')

# ---- MLD quantification (R1 comments 9, 10)
Q['mld_quant'] = grab(
    'The reduction is substantial rather than complete.',
    'across the open gyres.')

# ---- Fig 1 i-l density panels (R1 comment 11, 12)
Q['density_panels'] = grab(
    'The surface density anomalies',
    'outweighs cooling.')

# ---- Ross sector (R1 comment 13)
Q['ross'] = grab(
    'In the glacial states the sea ice response',
    'in the variability analysis below.')

# ---- thermal quantification (R1 comment 14)
Q['thermal_quant'] = grab(
    'This transformation is dominated by surface heat loss',
    'in the Discussion.')

# ---- density tendency definitions (R1 comment 19)
Q['tendency_defn'] = grab(
    'We refer throughout to the thermal density tendency',
    'freshwater and reduces it.')

# ---- convection location (R1 comment 23)
Q['convection'] = grab(
    'In PI this convection is concentrated in the Weddell sector',
    'over the continental shelf.')

# ---- SAM mechanism expansion (R1 comment 24)
Q['sam_mech'] = grab(
    'These composites indicate that the mode acts',
    'the transformation response follows both.')

# ---- glacial quantification (R1 comment 27)
Q['glacial_quant'] = grab(
    'The winter transformation maximum weakens to below 60',
    'concentrated in coastal regions.')

# ---- SAM repetition fix and Ross caveat (R1 comment 28)
Q['sam_nonuniform'] = grab(
    'This circumpolar picture does not hold uniformly',
    'sectors contribute little.')

# ---- sediment caveat (R1 comment 29, R3 L339)
Q['sediment'] = grab(
    'Whether this persistent coupling could be exploited',
    'Southern Annular Mode reconstruction.')

# ---- methods components (R1 comment 30)
Q['components'] = grab(
    'In the figures the radiative component denotes',
    'sensible heat fluxes.')

# ---- boundary conditions / GLAC1D (R1 general, comment 32)
Q['glac1d'] = grab(
    'We perform five equilibrium simulations spanning distinct climate states',
    'basal melt beneath floating ice shelves is not represented.')

# ---- spin-up and drift (R3 equilibrium)
Q['drift'] = grab(
    'Residual drift over the analysis period',
    'other paleoclimate states.')

# ---- previous validation (R1 general)
Q['prev_validation'] = grab(
    'Because the balance of agreement and disagreement in those evaluations',
    'on which the present analysis depends.')

# ---- Sidorenko evaluation (R1 general, R3)
Q['sidorenko'] = grab(
    'Independent evaluations of this model provide context',
    'reported here is our own.')

# ---- domain and density-class resolution (R2 major 1, R3)
Q['domain_test'] = grab(
    'Two factors contribute, and they can be separated.',
    'from 0.86 to 1.84~Sv')

Q['density_class'] = grab(
    'The second factor is the density range',
    'to bottom water proper.')

Q['alpha_check'] = grab(
    'We can exclude one candidate explanation',
    'documented above.')

# ---- Chen discussion (R2 major 2)
Q['chen_discussion'] = grab(
    'Our conclusions about the glacial state are consistent',
    'that Chen et al. proposed.')

# ---- Lhardy / glacial sea ice (R2 minor)
Q['lhardy'] = grab(
    'The ability of the model to maintain an extensive glacial',
    'quantitative agreement remains uncertain.')

# ---- westerly bias (R3)
Q['westerly'] = grab(
    'The simulated westerlies strengthen in both the glacial',
    'in the Discussion.')

Q['westerly_disc'] = grab(
    'A related limitation concerns the simulated glacial winds.',
    'wind-driven variability.')

# ---- MOC (R3)
Q['moc'] = grab(
    'The surface transformation changes described above',
    'wholesale collapse of the overturning.')

# ---- global age (R3)
Q['age_global'] = grab(
    'Because a single depth level gives an incomplete picture',
    'while the upper cell is maintained.')

# ---- polynya resolution (R2 minor)
Q['polynya'] = grab(
    'At this resolution the larger coastal polynyas',
    'appear blank in the map figures.')

# ---- ice sheet meltwater (R3 L195)
Q['meltwater'] = grab(
    'We note that this statement applies to the freshwater sources',
    'dominated by sea ice thermodynamics.')

# deep-ocean drift paragraph removed from the Methods at Xiaoxu's request,
# so there is nothing left to quote for it

# ---- SAM push-pull (R3)
Q['pushpull'] = grab(
    'An asymmetry between the two routes emerges',
    'operating through a different pathway.')

# ---- moderated SAM proxy claim (R1 G3)
Q['sam_proxy'] = grab(
    'Whether this persistent coupling could be exploited for proxy reconstruction',
    'marine Southern Annular Mode reconstruction.')

# ---- SAM brine mechanism as a sequence (R1 comment 24)
Q['brine_mech'] = grab(
    'This pattern is controlled by sea ice processes, and the chain',
    'more generally \\cite{Abernathey2016}.')

# ---- panel-level references example (R1 comment 21)
Q['panel_refs'] = grab(
    'In PI, strong thermal densification (red shading)',
    'buoyancy budgets for the Southern Ocean \\cite{Cerovecki2013,Abernathey2016}.')

missing = [k for k, v in Q.items() if not v]
for k in missing:
    print('NOT FOUND:', k)
Q = {k: v for k, v in Q.items() if v}

OUT.write_text(json.dumps(Q, indent=1, ensure_ascii=False))
print(f'\nextracted {len(Q)} verbatim passages -> {OUT}')
for k, v in Q.items():
    print(f'  {k:18s} {len(v):5d} chars  {v[:70]}...')
