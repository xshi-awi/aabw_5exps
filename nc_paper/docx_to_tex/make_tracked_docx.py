#!/usr/bin/env python3
"""Tracked-changes Word file: original submission -> revised manuscript (user 11-27 13:38).

Inputs are the two manuscripts converted by tex2docx (orig.docx, rev.docx). The output keeps the revised package
(styles, figures, numbering) and rewrites the body with Word revision marks (w:ins / w:del, author AUTHOR):
- body blocks (paragraphs and tables) are aligned with difflib on their text; equal blocks are kept as revised;
- in a replaced stretch, blocks are paired by similarity; paired paragraphs get a word-level diff, unpaired ones
  become whole inserted or deleted paragraphs (paragraph mark marked too); changed tables are deleted and inserted;
- figures are the revised figures; figures with no counterpart are tracked as deleted (original image) or inserted;
  a figure kept in both is shown unmarked with the revised image (its caption is diffed as text);
- citation and cross-reference hyperlinks and bookmarks are flattened to plain runs (same text);
- a changed maths object: the original object marked deleted and the revised one marked inserted, the way Word
  marks changes inside equations (each maths run in w:ins/w:del, the mark also in every structure's m:ctrlPr).
  LibreOffice 7.6 does not show marks inside equations; Word does.
Also fixes the order m:scr before m:sty in maths run properties (pandoc writes it the other way round).
Usage: make_tracked_docx.py orig.docx rev.docx out.docx"""
import copy
import difflib
import re
import sys
import zipfile

from lxml import etree

ORIG, REV, OUT = sys.argv[1:4]
AUTHOR, DATE = 'Xiaoxu Shi', '2026-11-27T00:00:00Z'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
M = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
WP = 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
PR = 'http://schemas.openxmlformats.org/package/2006/relationships'
XS = '{http://www.w3.org/XML/1998/namespace}space'
q = lambda t, ns=W: '{%s}%s' % (ns, t)
_id = [90000]


def rid():
    _id[0] += 1
    return str(_id[0])


def mark(tag):
    e = etree.Element(q(tag))
    e.set(q('id'), rid())
    e.set(q('author'), AUTHOR)
    e.set(q('date'), DATE)
    return e


def load(path):
    z = zipfile.ZipFile(path)
    return z, etree.fromstring(z.read('word/document.xml'))


zo, xo = load(ORIG)
zr, xr = load(REV)
bo, br = xo.find(q('body')), xr.find(q('body'))
rels_r = etree.fromstring(zr.read('word/_rels/document.xml.rels'))
rels_o = etree.fromstring(zo.read('word/_rels/document.xml.rels'))
orig_rel = {e.get('Id'): e.get('Target') for e in rels_o}
new_media = {}          # target name in output -> bytes
rel_map = {}            # original rId -> new rId in output


def text_of(el):
    return ''.join(t.text or '' for t in el.iter(q('t'), q('delText'), q('t', M)))


def norm(s):
    return re.sub(r'\s+', ' ', s).strip()


SIG = {}


def sig(el):
    return SIG[el]


def blocks(body):
    """Body blocks; their signature for the alignment. A figure paragraph has no text: it is keyed on the
    caption that follows it, so a figure pairs with the same figure and a moved figure shows as deleted
    and inserted."""
    bl = [e for e in body if e.tag in (q('p'), q('tbl'))]
    for k, el in enumerate(bl):
        if el.tag == q('tbl'):
            SIG[el] = 'TBL: ' + norm(text_of(el))
        elif el.find('.//' + q('drawing')) is not None:
            cap = norm(text_of(bl[k + 1])) if k + 1 < len(bl) else ''
            SIG[el] = 'IMG ' + re.sub(r'^Fig\. [A-Z]?\d+\s*', '', cap)
        else:
            SIG[el] = norm(text_of(el))
    return bl


# ------------------------------------------------------------------ tokens
def tokens(p):
    """[(key, kind, payload, rPr)] in document order; hyperlinks flattened, bookmarks dropped."""
    out = []

    def run_tokens(r):
        rpr = r.find(q('rPr'))
        for c in r:
            if c.tag == q('t'):
                for w in re.findall(r'\s+|[^\s]+', c.text or ''):
                    out.append((w, 'text', w, rpr))
            elif c.tag == q('tab'):
                out.append(('\t', 'tab', None, rpr))
            elif c.tag == q('drawing'):
                out.append(('IMG', 'drawing', c, rpr))
            elif c.tag == q('br'):
                out.append(('\n', 'br', None, rpr))
    for c in p:
        if c.tag == q('r'):
            run_tokens(c)
        elif c.tag == q('hyperlink'):
            for r in c.iter(q('r')):
                run_tokens(r)
        elif c.tag in (q('oMath', M), q('oMathPara', M)):
            out.append(('MATH:' + norm(text_of(c)), 'math', c, None))
    return out


def rpr_key(rpr):
    return etree.tostring(rpr) if rpr is not None else b''


def emit(p, toks, status, src):
    """Append runs for tokens with one status ('eq', 'ins', 'del') to paragraph p. src: 'o' or 'r'."""
    group = []

    def flush():
        if not group:
            return
        rpr = group[0][3]
        r = etree.Element(q('r'))
        if rpr is not None:
            r.append(copy.deepcopy(rpr))
        buf = ''
        for key, kind, payload, _ in group:
            if kind == 'text':
                buf += payload
            else:
                if buf:
                    t = etree.SubElement(r, q('delText' if status == 'del' else 't'))
                    t.text, buf = buf, ''
                    t.set(XS, 'preserve')
                etree.SubElement(r, q('tab' if kind == 'tab' else 'br'))
        if buf:
            t = etree.SubElement(r, q('delText' if status == 'del' else 't'))
            t.text = buf
            t.set(XS, 'preserve')
        wrap(p, r, status)
        group.clear()

    for tk in toks:
        key, kind, payload, rpr = tk
        if kind in ('math', 'drawing'):
            flush()
            if kind == 'math':
                p.append(track_math(payload, status))
            else:
                r = etree.Element(q('r'))
                if rpr is not None:
                    r.append(copy.deepcopy(rpr))
                d = copy.deepcopy(payload)
                if src == 'o':
                    remap_drawing(d)
                r.append(d)
                wrap(p, r, status)
            continue
        if group and (rpr_key(group[0][3]) != rpr_key(rpr)):
            flush()
        group.append(tk)
    flush()


MSTRUCT = ['acc', 'bar', 'box', 'borderBox', 'd', 'eqArr', 'f', 'func', 'groupChr', 'limLow', 'limUpp', 'm',
           'nary', 'phant', 'rad', 'sPre', 'sSub', 'sSubSup', 'sSup']


def track_math(om, status):
    """A maths object marked the way Word marks changes inside an equation: every maths run wrapped in
    w:ins/w:del, and the control properties of every structure (fraction, subscript, ...) carry the mark."""
    om = copy.deepcopy(om)
    if status == 'eq':
        return om
    tag = 'ins' if status == 'ins' else 'del'
    for r in list(om.iter(q('r', M))):
        m_ = mark(tag)
        r.addprevious(m_)
        m_.append(r)
    for st in list(om.iter(*[q(s, M) for s in MSTRUCT])):
        name = q(etree.QName(st).localname + 'Pr', M)
        pr = st.find(name)
        if pr is None:
            pr = etree.Element(name)
            st.insert(0, pr)
        cp = pr.find(q('ctrlPr', M))
        if cp is None:
            cp = etree.SubElement(pr, q('ctrlPr', M))
        rpr = cp.find(q('rPr'))
        m_ = mark(tag)
        if rpr is not None:
            cp.remove(rpr)
        else:
            rpr = etree.Element(q('rPr'))
        m_.append(rpr)
        for c in list(cp):
            cp.remove(c)
        cp.append(m_)
    return om


def wrap(p, r, status):
    if status == 'eq':
        p.append(r)
    else:
        m = mark('ins' if status == 'ins' else 'del')
        m.append(r)
        p.append(m)


def remap_drawing(d):
    """A drawing copied from the original: bring its image into the output package under a new rId."""
    for blip in d.iter(q('blip', A)):
        old = blip.get(q('embed', R))
        if old not in rel_map:
            target = orig_rel[old]
            data = zo.read('word/' + target)
            name = 'media/orig_' + target.split('/')[-1]
            new_media['word/' + name] = data
            nid = 'rIdOrig%d' % (len(rel_map) + 1)
            e = etree.SubElement(rels_r, '{%s}Relationship' % PR)
            e.set('Id', nid)
            e.set('Type', 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/image')
            e.set('Target', name)
            rel_map[old] = nid
        blip.set(q('embed', R), rel_map[old])


def mark_paragraph(p, status):
    ppr = p.find(q('pPr'))
    if ppr is None:
        ppr = etree.Element(q('pPr'))
        p.insert(0, ppr)
    rpr = ppr.find(q('rPr'))
    if rpr is None:
        rpr = etree.Element(q('rPr'))
        # rPr goes after every other pPr child except sectPr and pPrChange
        tail = [c for c in ppr if c.tag in (q('sectPr'), q('pPrChange'))]
        if tail:
            tail[0].addprevious(rpr)
        else:
            ppr.append(rpr)
    rpr.insert(0, mark('ins' if status == 'ins' else 'del'))


def new_par(src_p, keep_bookmarks=False):
    p = etree.Element(q('p'))
    ppr = src_p.find(q('pPr'))
    if ppr is not None:
        p.append(copy.deepcopy(ppr))
    if keep_bookmarks:                                    # revised bookmarks (reference targets of citations)
        for b in src_p.iter(q('bookmarkStart')):
            p.append(copy.deepcopy(b))
            e = etree.SubElement(p, q('bookmarkEnd'))
            e.set(q('id'), b.get(q('id')))
    return p


def diff_par(po, pr):
    to, tr = tokens(po), tokens(pr)
    p = new_par(pr, keep_bookmarks=True)
    sm = difflib.SequenceMatcher(None, [t[0] for t in to], [t[0] for t in tr], autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == 'equal':
            emit(p, tr[j1:j2], 'eq', 'r')
        else:
            if i2 > i1:
                emit(p, to[i1:i2], 'del', 'o')
            if j2 > j1:
                emit(p, tr[j1:j2], 'ins', 'r')
    return p


def whole_par(src, status):
    p = new_par(src, keep_bookmarks=(status == 'ins'))
    emit(p, tokens(src), status, 'o' if status == 'del' else 'r')
    mark_paragraph(p, status)
    return p


def whole_tbl(src, status):
    t = copy.deepcopy(src)
    for tr_ in t.iter(q('tr')):
        trpr = tr_.find(q('trPr'))
        if trpr is None:
            trpr = etree.Element(q('trPr'))
            tblpr = tr_.find(q('tblPrEx'))
            (tblpr.addnext if tblpr is not None else lambda e: tr_.insert(0, e))(trpr)
        trpr.append(mark('ins' if status == 'ins' else 'del'))
    for p in list(t.iter(q('p'))):
        np_ = whole_par(p, status)
        p.getparent().replace(p, np_)
    return t


def similar(a, b):
    return difflib.SequenceMatcher(None, sig(a).split(), sig(b).split(), autojunk=False).ratio()


# ------------------------------------------------------------------ body
BO, BR = blocks(bo), blocks(br)
out = []
stats = dict(equal=0, diffed=0, ins_par=0, del_par=0, ins_tbl=0, del_tbl=0)
sm = difflib.SequenceMatcher(None, [sig(e) for e in BO], [sig(e) for e in BR], autojunk=False)
for op, i1, i2, j1, j2 in sm.get_opcodes():
    if op == 'equal':
        for e in BR[j1:j2]:
            out.append(copy.deepcopy(e))
            stats['equal'] += 1
        continue
    O_, R_ = BO[i1:i2], BR[j1:j2]
    # pair blocks in order: the monotone pairing with the largest summed similarity (pairs need >= 0.45)
    no, nr = len(O_), len(R_)
    S = [[0.0] * nr for _ in range(no)]
    for a in range(no):
        wa = sig(O_[a]).split()
        for b in range(nr):
            if O_[a].tag != R_[b].tag or (sig(O_[a]).startswith('IMG') != sig(R_[b]).startswith('IMG')):
                continue
            m_ = difflib.SequenceMatcher(None, wa, sig(R_[b]).split(), autojunk=False)
            if m_.quick_ratio() >= 0.45:
                s = m_.ratio()
                S[a][b] = s if s >= 0.45 else 0.0
    # only mutual best matches may pair: figure captions share most words, and two figures that swapped
    # places must not be paired with each other's caption (shown as deleted and inserted instead)
    if no and nr:
        rb = [max(range(nr), key=lambda b: S[a][b]) for a in range(no)]
        ca = [max(range(no), key=lambda a: S[a][b]) for b in range(nr)]
        for a in range(no):
            for b in range(nr):
                if S[a][b] and not (rb[a] == b and ca[b] == a):
                    S[a][b] = 0.0
    V = [[0.0] * (nr + 1) for _ in range(no + 1)]
    for a in range(1, no + 1):
        for b in range(1, nr + 1):
            V[a][b] = max(V[a - 1][b], V[a][b - 1], V[a - 1][b - 1] + S[a - 1][b - 1] if S[a - 1][b - 1] else 0.0)
    pairs, a, b = [], no, nr
    while a > 0 and b > 0:
        if S[a - 1][b - 1] and V[a][b] == V[a - 1][b - 1] + S[a - 1][b - 1]:
            pairs.append((a - 1, b - 1))
            a, b = a - 1, b - 1
        elif V[a][b] == V[a - 1][b]:
            a -= 1
        else:
            b -= 1
    pairs.reverse()
    pa = dict(pairs)
    pb = {b: a for a, b in pairs}
    a = 0
    for b in range(len(R_)):
        if b in pb:
            while a < pb[b]:                      # original blocks before this pair: deleted
                if a not in pa:
                    out.append(whole_par(O_[a], 'del') if O_[a].tag == q('p') else whole_tbl(O_[a], 'del'))
                    stats['del_par' if O_[a].tag == q('p') else 'del_tbl'] += 1
                a += 1
            if R_[b].tag == q('p'):
                out.append(diff_par(O_[pb[b]], R_[b]))
                stats['diffed'] += 1
            else:
                out.append(whole_tbl(O_[pb[b]], 'del'))
                out.append(whole_tbl(R_[b], 'ins'))
                stats['del_tbl'] += 1
                stats['ins_tbl'] += 1
            a = pb[b] + 1
        else:
            out.append(whole_par(R_[b], 'ins') if R_[b].tag == q('p') else whole_tbl(R_[b], 'ins'))
            stats['ins_par' if R_[b].tag == q('p') else 'ins_tbl'] += 1
    while a < len(O_):
        if a not in pa:
            out.append(whole_par(O_[a], 'del') if O_[a].tag == q('p') else whole_tbl(O_[a], 'del'))
            stats['del_par' if O_[a].tag == q('p') else 'del_tbl'] += 1
        a += 1

sect = br.find(q('sectPr'))
for e in list(br):
    br.remove(e)
for e in out:
    br.append(e)
if sect is not None:
    br.append(sect)

# unique drawing ids
for k, d in enumerate(xr.iter(q('docPr', WP))):
    d.set('id', str(1000 + k))
# maths run properties: m:scr must come before m:sty
for mrpr in xr.iter(q('rPr', M)):
    order = ['lit', 'nor', 'scr', 'sty', 'brk', 'aln']
    kids = sorted(mrpr, key=lambda c: order.index(etree.QName(c).localname) if etree.QName(c).localname in order else 9)
    for c in kids:
        mrpr.remove(c)
        mrpr.append(c)

# ------------------------------------------------------------------ write
ct = etree.fromstring(zr.read('[Content_Types].xml'))
exts = {e.get('Extension') for e in ct if e.get('Extension')}
for name in new_media:
    ext = name.rsplit('.', 1)[-1].lower()
    if ext not in exts:
        e = etree.SubElement(ct, '{http://schemas.openxmlformats.org/package/2006/content-types}Default')
        e.set('Extension', ext)
        e.set('ContentType', {'png': 'image/png', 'jpeg': 'image/jpeg', 'jpg': 'image/jpeg', 'emf': 'image/x-emf'}.get(ext, 'application/octet-stream'))
        exts.add(ext)
with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as z:
    for info in zr.infolist():
        if info.filename.endswith('/'):
            continue
        if info.filename == 'word/document.xml':
            data = etree.tostring(xr, xml_declaration=True, encoding='UTF-8', standalone=True)
        elif info.filename == 'word/_rels/document.xml.rels':
            data = etree.tostring(rels_r, xml_declaration=True, encoding='UTF-8', standalone=True)
        elif info.filename == '[Content_Types].xml':
            data = etree.tostring(ct, xml_declaration=True, encoding='UTF-8', standalone=True)
        else:
            data = zr.read(info.filename)
        z.writestr(info, data)
    for name, data in new_media.items():
        z.writestr(name, data)
print('blocks', stats, 'original images carried over', len(rel_map), '->', OUT)
