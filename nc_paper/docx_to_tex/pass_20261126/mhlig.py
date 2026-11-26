"""Spell out MH and LIG in running text (Reviewer 3 minor comment, user 2026-11-26:
「正文里面全部用全称，只在图片里面用简称」). Shared by the manuscript and the letter scripts so
that quoted sentences change identically in both. Ordered from specific to generic.
Kept on purpose: the label definitions (Results 2.1, Methods list), Table 1 rows, LaTeX comments.
"""
import re

M, G = 'mid-Holocene', 'Last Interglacial'
RULES = [
    # definitions
    (r'Throughout the Results we refer to the five simulated climate states by the abbreviations used in the figures: PI for pre-industrial, MH for mid-Holocene, LIG for the last interglacial, LGM for the Last Glacial Maximum, and MIS3 for Marine Isotope Stage 3.',
     'The figures label the five simulated climate states as PI (pre-industrial), MH (mid-Holocene), LIG (Last Interglacial), LGM (Last Glacial Maximum) and MIS3 (Marine Isotope Stage 3). The text uses the full names for the mid-Holocene and the Last Interglacial.'),
    (r'mid-Holocene (MH), and last interglacial (LIG)', 'mid-Holocene (MH), and Last Interglacial (LIG)'),
    # captions with an old grammar slip
    (r'anomalies in MH, LIG, LGM, and MIS3 in relative to PI', f'anomalies in the {M}, the {G}, LGM, and MIS3 relative to PI'),
    # lists
    (r'(PI, MH, LIG', f'(PI, {M}, {G}'),
    (r'PI, MH, LIG, LGM', f'PI, the {M}, the {G}, LGM'),
    (r'PI, MH, and LIG', f'PI, the {M} and the {G}'),
    (r'PI, MH and LIG', f'PI, the {M} and the {G}'),
    (r'(MH, LIG', f'({M}, {G}'),
    (r'MH, LIG, LGM', f'the {M}, the {G}, LGM'),
    (r'The MH and LIG simulations', f'The {M} and {G} simulations'),
    (r'MH and LIG', f'the {M} and the {G}'),
    (r'The LIG and MIS3 values', f'The {G} and MIS3 values'),
    (r'PI and MH winter', f'PI and {M} winter'),
    # panel letters
    (r'(b) MH, (c) LIG', f'(b) {M}, (c) {G}'),
    (r'(b,g) MH, (c,h) LIG', f'(b,g) {M}, (c,h) {G}'),
    (r'(f) MH minus PI, (g) LIG minus PI', f'(f) {M} minus PI, (g) {G} minus PI'),
    # sentence starts
    (r'The MH ', f'The {M} '), (r'The LIG ', f'The {G} '),
    (r'MH exhibits', f'The {M} exhibits'), (r'LIG exhibits', f'The {G} exhibits'),
    (r'MH anomalies', 'Mid-Holocene anomalies'), (r'LIG anomalies', f'{G} anomalies'),
    (r'That MH ', f'That the {M} '),
    (r'In MH', f'In the {M}'), (r'In LIG', f'In the {G}'),
    (r'During LIG', f'During the {G}'),
    # prepositions
    (r'in MH', f'in the {M}'), (r'in LIG', f'in the {G}'),
    (r'during MH', f'during the {M}'), (r'during LIG', f'during the {G}'),
    (r'for MH', f'for the {M}'), (r'for LIG', f'for the {G}'),
]


def spell_out(text):
    for old, new in RULES:
        text = re.sub(re.escape(old) + (r'\b' if old[-1].isalnum() else ''), lambda m: new, text)
    return text


def apply_to_tex(T):
    """Apply to non-comment lines; leave table rows starting with 'MH ' or 'LIG ' alone."""
    out = []
    for line in T.split('\n'):
        if line.lstrip().startswith('%') or re.match(r'^(MH|LIG)\s*&', line):
            out.append(line)
            continue
        m = re.search(r'(?<!\\)%', line)   # protect a trailing LaTeX comment (unescaped %)
        if m:
            out.append(spell_out(line[:m.start()]) + line[m.start():])
        else:
            out.append(spell_out(line))
    return '\n'.join(out)


def leftovers(T):
    bad = []
    for i, line in enumerate(T.split('\n'), 1):
        if line.lstrip().startswith('%') or re.match(r'^(MH|LIG)\s*&', line):
            continue
        m = re.search(r'(?<!\\)%', line)
        body = line[:m.start()] if m else line
        body = body.replace('MH (mid-Holocene), LIG (Last Interglacial)', '').replace('mid-Holocene (MH), and Last Interglacial (LIG)', '')
        for w in re.finditer(r'\b(MH|LIG)\b', body):
            bad.append((i, body[max(0, w.start() - 40):w.end() + 30]))
    return bad
