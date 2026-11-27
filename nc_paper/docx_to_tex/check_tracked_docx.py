#!/usr/bin/env python3
"""Check a tracked-changes docx made by make_tracked_docx.py.
Accept all changes -> text must equal the revised docx; reject all -> text must equal the original docx.
Text is compared paragraph by paragraph (w:t, w:delText and maths m:t, whitespace collapsed; hyperlinks and
bookmarks ignored). Also counts the marks per author and the figures in each view.
Usage: check_tracked_docx.py tracked.docx orig.docx rev.docx"""
import copy
import difflib
import re
import sys
import zipfile

from lxml import etree

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
M = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
q = lambda t, ns=W: '{%s}%s' % (ns, t)


def body(path):
    return etree.fromstring(zipfile.ZipFile(path).read('word/document.xml')).find(q('body'))


def apply(b, accept):
    """accept=True: drop w:del, unwrap w:ins; False: drop w:ins, unwrap w:del. Paragraphs whose mark is
    removed are merged into the next paragraph."""
    b = copy.deepcopy(b)
    drop, keep = ('del', 'ins') if accept else ('ins', 'del')
    # paragraph marks and table rows first
    for p in list(b.iter(q('p'))):
        mk = p.find(q('pPr') + '/' + q('rPr') + '/' + q(drop))
        if mk is not None:
            p.set('merge', '1')
    for tr in list(b.iter(q('tr'))):
        if tr.find(q('trPr') + '/' + q(drop)) is not None:
            tr.getparent().remove(tr)
    for e in list(b.iter(q(drop))):
        par = e.getparent()
        if par is not None and par.tag != q('rPr') and par.tag != q('trPr'):
            par.remove(e)
    for e in list(b.iter(q(keep))):
        par = e.getparent()
        if par is None or par.tag in (q('rPr'), q('trPr')):
            continue
        i = list(par).index(e)
        for c in reversed(list(e)):
            par.insert(i, c)
        par.remove(e)
    for t in b.iter(q('delText')):
        t.tag = q('t')
    # merge paragraphs whose mark was removed into the following paragraph
    for p in reversed(list(b.iter(q('p')))):
        if p.get('merge') != '1':
            continue
        par = p.getparent()
        sib = p.getnext()
        content = [c for c in p if c.tag != q('pPr')]
        if sib is not None and sib.tag == q('p'):
            at = 1 if len(sib) and sib[0].tag == q('pPr') else 0
            for c in reversed(content):
                sib.insert(at, c)
            par.remove(p)
        elif not ''.join(x.text or '' for x in p.iter(q('t'), q('t', M))).strip():
            par.remove(p)
    for t in b.iter(q('tbl')):
        if t.find('.//' + q('tr')) is None:
            t.getparent().remove(t)
    return b


def paras(b):
    out = []
    for p in b.iter(q('p')):
        s = ''.join(x.text or '' for x in p.iter(q('t'), q('delText'), q('t', M)))
        s = re.sub(r'\s+', ' ', s).strip()
        img = len(p.findall('.//' + q('drawing')))
        if s or img:
            out.append(('[IMG] ' * img) + s)
    return out


def compare(name, got, ref):
    if got == ref:
        print(f'{name}: identical ({len(ref)} paragraphs)')
        return True
    print(f'{name}: DIFFERENT ({len(got)} vs {len(ref)} paragraphs)')
    for line in list(difflib.unified_diff(ref, got, 'expected', 'got', n=0, lineterm=''))[:30]:
        print('   ', line[:300])
    return False


T, O, R = sys.argv[1:4]
bt = body(T)
authors = {}
for tag in ('ins', 'del'):
    for e in bt.iter(q(tag)):
        authors[(tag, e.get(q('author')))] = authors.get((tag, e.get(q('author'))), 0) + 1
print('marks:', authors)
print('maths objects with inserted/deleted runs:', sum(1 for o in bt.iter(q('oMath', M)) if o.find('.//' + q('ins')) is not None),
      sum(1 for o in bt.iter(q('oMath', M)) if o.find('.//' + q('del')) is not None))
acc, rej = apply(bt, True), apply(bt, False)
ok1 = compare('accept all = revised', paras(acc), paras(body(R)))
ok2 = compare('reject all = original', paras(rej), paras(body(O)))
for nm, b in (('accepted', acc), ('rejected', rej), ('revised', body(R)), ('original', body(O))):
    print(f'  figures in {nm}: {len(b.findall(".//" + q("drawing")))}')
sys.exit(0 if ok1 and ok2 else 1)
