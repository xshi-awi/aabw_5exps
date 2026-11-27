#!/usr/bin/env python3
"""Compare the Word version of the letter (rendered to PDF by LibreOffice) with the LaTeX PDF.
Usage: python3 compare_word_pdf.py <latex.pdf> <word_render.pdf> <letter.docx> <letter_figs dir>
Reports: word-level text differences, colour class of every aligned word (black / blue / magenta),
italic quote words, figure order (by md5 of the embedded PNGs), figure and table numbers."""
import collections
import difflib
import hashlib
import re
import sys
import zipfile

import pymupdf as fitz

TEX_PDF, WORD_PDF, DOCX, FIGDIR = sys.argv[1:5]


def cls(color):
    r, g, b = (color >> 16) & 255, (color >> 8) & 255, color & 255
    if r > 150 and b > 80 and g < 80:
        return 'magenta'
    if b > 100 and r < 60:
        return 'blue'
    return 'black'


def words(pdf):
    out = []
    for page in fitz.open(pdf):
        H = page.rect.height
        for b in page.get_text('dict')['blocks']:
            for l in b.get('lines', []):
                for s in l['spans']:
                    if s['bbox'][1] > 0.92 * H:      # page number in the footer
                        continue
                    it = 'Italic' in s['font'] or bool(s['flags'] & 2)
                    for w in re.findall(r'\S+', s['text']):
                        out.append((w, cls(s['color']), it, page.number + 1))
    return out


def norm(w):
    w = w.replace('ﬁ', 'fi').replace('ﬂ', 'fl').replace('ﬀ', 'ff').replace('ﬃ', 'ffi').replace('’', "'").replace('‘', "'")
    w = w.replace('“', '"').replace('”', '"').replace('−', '-').replace('–', '-').replace('—', '-')
    return w


A, B = words(TEX_PDF), words(WORD_PDF)
# join words hyphenated across lines in the LaTeX PDF
def dehyph(ws):
    out = []
    for w in ws:
        if out and out[-1][0].endswith('\xad'):               # LibreOffice soft hyphen at a line end
            prev = out.pop()
            out.append((prev[0][:-1] + w[0], prev[1], prev[2], prev[3]))
        elif out and out[-1][0].endswith('-') and len(out[-1][0]) > 1 and w[0][:1].islower():
            prev = out.pop()
            out.append((prev[0][:-1] + w[0], prev[1], prev[2], prev[3]))
        else:
            out.append(w)
    return out
A, B = dehyph(A), dehyph(B)
a, b = [norm(x[0]) for x in A], [norm(x[0]) for x in B]
sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
print('words: latex %d, word %d, ratio %.4f' % (len(a), len(b), sm.ratio()))
ndiff = 0
for op, i1, i2, j1, j2 in sm.get_opcodes():
    if op == 'equal':
        continue
    ndiff += 1
    if ndiff <= 60:
        print('  %-7s p%-3s latex: %-60s | word p%s: %s' % (op, A[i1][3] if i1 < len(A) else '-', ' '.join(a[i1:i2])[:60],
                                                       B[j1][3] if j1 < len(B) else '-', ' '.join(b[j1:j2])[:60]))
print('text blocks that differ:', ndiff)

# colour and italic agreement on the aligned words
cc = collections.Counter()
bad = []
for op, i1, i2, j1, j2 in sm.get_opcodes():
    if op != 'equal':
        continue
    for k in range(i2 - i1):
        x, y = A[i1 + k], B[j1 + k]
        cc[(x[1], y[1])] += 1
        if x[1] != y[1] or x[2] != y[2]:
            bad.append((x[3], y[3], x[0], x[1], y[1], x[2], y[2]))
print('colour pairs (latex, word):', dict(cc))
print('aligned words whose colour or italic differ: %d' % len(bad))
runs = []
for t in bad:
    if runs and runs[-1][-1][0] == t[0] and runs[-1][-1][3:] == t[3:]:
        runs[-1].append(t)
    else:
        runs.append([t])
for r in runs[:40]:
    t = r[0]
    print('  latex p%s word p%s %-8s->%-8s italic %s->%s: %s' % (t[0], t[1], t[3], t[4], t[5], t[6], ' '.join(x[2] for x in r)[:90]))

# figures
md5 = lambda d: hashlib.md5(d).hexdigest()
import os
lf = {md5(open(os.path.join(FIGDIR, f), 'rb').read()): f for f in os.listdir(FIGDIR) if f.endswith('.png')}
with zipfile.ZipFile(DOCX) as z:
    doc = z.read('word/document.xml').decode('utf8')
    rels = z.read('word/_rels/document.xml.rels').decode('utf8')
    rid = dict(re.findall(r'Id="([^"]+)"[^>]*Target="([^"]+)"', rels))
    order = re.findall(r'r:embed="([^"]+)"', doc)
    figs = [lf.get(md5(z.read('word/' + rid[r])), '??' + rid[r]) for r in order]
tex = open(os.path.join(os.path.dirname(FIGDIR.rstrip('/')), 'response_new.tex'), encoding='utf8').read()
tex_figs = re.findall(r'\\rfig\{[^}]*\}\{\.\./letter_figs/([^}]+)\}', tex)
print('figures: latex %d, word %d, same order and files: %s' % (len(tex_figs), len(figs), tex_figs == figs))
if tex_figs != figs:
    for k, (x, y) in enumerate(zip(tex_figs, figs)):
        print('   ', k + 1, x, y, '' if x == y else '<-- differs')
caps_t = re.findall(r'Figure (R\d+)\.', ' '.join(x[0] for x in A))
caps_w = re.findall(r'Figure (R\d+)\.', ' '.join(x[0] for x in B))
print('caption labels latex', caps_t)
print('caption labels word ', caps_w)
refs_t = collections.Counter(re.findall(r'(?:Figure|Fig\.|Table) (R\d+)', ' '.join(x[0] for x in A)))
refs_w = collections.Counter(re.findall(r'(?:Figure|Fig\.|Table) (R\d+)', ' '.join(x[0] for x in B)))
print('R-number mentions equal:', refs_t == refs_w, '' if refs_t == refs_w else (refs_t - refs_w, refs_w - refs_t))
