"""Dump a docx into a readable, run-level listing (untrusted input: parse only, never execute).

Usage: python3 -I dump_docx.py <file.docx> <out.json> <out.txt>
Each paragraph is a list of runs with flags: b(old) i(talic) s(trike) h(ighlight colour) c(olour)
sup/sub, plus markers for comment ranges, images, footnote references, tracked ins/del.
"""
import html
import json
import re
import sys
import zipfile

W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
from xml.etree import ElementTree as ET

z = zipfile.ZipFile(sys.argv[1])
doc = ET.fromstring(z.read('word/document.xml'))
rels = ET.fromstring(z.read('word/_rels/document.xml.rels'))
relmap = {r.get('Id'): r.get('Target') for r in rels}
comments = {}
if 'word/comments.xml' in z.namelist():
    cx = ET.fromstring(z.read('word/comments.xml'))
    for c in cx.findall(W + 'comment'):
        txt = '\n'.join(''.join(t.text or '' for t in p.iter(W + 't')) for p in c.findall(W + 'p'))
        comments[c.get(W + 'id')] = dict(author=c.get(W + 'author'), date=c.get(W + 'date'), text=txt)

body = doc.find(W + 'body')
paras = []


def runprops(r):
    rp = r.find(W + 'rPr')
    f = {}
    if rp is None:
        return f
    for tag, key in (('b', 'b'), ('i', 'i'), ('strike', 's'), ('dstrike', 's')):
        e = rp.find(W + tag)
        if e is not None and e.get(W + 'val') not in ('0', 'false'):
            f[key] = 1
    h = rp.find(W + 'highlight')
    if h is not None and h.get(W + 'val') not in (None, 'none'):
        f['h'] = h.get(W + 'val')
    sh = rp.find(W + 'shd')
    if sh is not None and sh.get(W + 'fill') not in (None, 'auto', 'FFFFFF'):
        f['shd'] = sh.get(W + 'fill')
    c = rp.find(W + 'color')
    if c is not None and c.get(W + 'val') not in (None, 'auto', '000000'):
        f['c'] = c.get(W + 'val')
    v = rp.find(W + 'vertAlign')
    if v is not None:
        f['va'] = v.get(W + 'val')
    return f


def walk(el, out, ctx):
    for ch in el:
        tag = ch.tag.replace(W, '')
        if tag == 'r':
            f = runprops(ch)
            f.update(ctx)
            for x in ch:
                xt = x.tag.replace(W, '')
                if xt == 't':
                    out.append(dict(t=x.text or '', **f))
                elif xt == 'tab':
                    out.append(dict(t='\t', **f))
                elif xt in ('br', 'cr'):
                    out.append(dict(t='\n', br=1, **f))
                elif xt == 'footnoteReference':
                    out.append(dict(t='', fn=x.get(W + 'id')))
                elif xt == 'drawing' or xt == 'pict':
                    for b in x.iter():
                        rid = b.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed') \
                            or b.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
                        if rid and rid in relmap:
                            out.append(dict(t='', img=relmap[rid]))
                elif xt == 'sym':
                    out.append(dict(t='', sym=x.get(W + 'char'), font=x.get(W + 'font'), **f))
                elif xt == 'delText':
                    out.append(dict(t=x.text or '', deltext=1, **f))
        elif tag == 'commentRangeStart':
            out.append(dict(t='', cstart=ch.get(W + 'id')))
        elif tag == 'commentRangeEnd':
            out.append(dict(t='', cend=ch.get(W + 'id')))
        elif tag == 'ins':
            walk(ch, out, dict(ctx, ins=1))
        elif tag == 'del':
            walk(ch, out, dict(ctx, dele=1))
        elif tag in ('hyperlink', 'smartTag', 'sdt', 'sdtContent', 'fldSimple', 'customXml'):
            walk(ch, out, ctx)
        elif tag == 'r' or tag == 'pPr':
            pass


def para(p):
    runs = []
    walk(p, runs, {})
    ppr = p.find(W + 'pPr')
    style = None
    if ppr is not None:
        ps = ppr.find(W + 'pStyle')
        style = ps.get(W + 'val') if ps is not None else None
        rp = ppr.find(W + 'rPr')
    return dict(style=style, runs=runs)


for el in body:
    tag = el.tag.replace(W, '')
    if tag == 'p':
        paras.append(para(el))
    elif tag == 'tbl':
        for row in el.iter(W + 'tr'):
            cells = []
            for cell in row.iter(W + 'tc'):
                cr = []
                for p in cell.iter(W + 'p'):
                    cr.extend(para(p)['runs'] + [dict(t=' | ')])
                cells.append(cr)
            paras.append(dict(style='TABLEROW', runs=[r for c in cells for r in c]))

json.dump(dict(paras=paras, comments=comments), open(sys.argv[2], 'w'), ensure_ascii=False, indent=0)

with open(sys.argv[3], 'w') as o:
    for i, p in enumerate(paras):
        segs = []
        for r in p['runs']:
            if 'cstart' in r:
                segs.append('⟦C%s:' % r['cstart'])
                continue
            if 'cend' in r:
                segs.append(':C%s⟧' % r['cend'])
                continue
            if 'img' in r:
                segs.append('⟦IMG %s⟧' % r['img'])
                continue
            if 'fn' in r:
                segs.append('⟦FN%s⟧' % r['fn'])
                continue
            if 'sym' in r:
                segs.append('⟦SYM %s %s⟧' % (r['font'], r['sym']))
                continue
            t = r['t']
            tags = []
            if r.get('s'):
                tags.append('S')
            if r.get('h'):
                tags.append('H')
            if r.get('dele') or r.get('deltext'):
                tags.append('DEL')
            if r.get('ins'):
                tags.append('INS')
            if r.get('va'):
                tags.append(r['va'][:3].upper())
            if tags:
                segs.append('«%s:%s»' % (','.join(tags), t))
            else:
                segs.append(t)
        line = ''.join(segs)
        flags = set()
        for r in p['runs']:
            if r.get('b'):
                flags.add('b')
            if r.get('i'):
                flags.add('i')
            if r.get('c'):
                flags.add('c' + r['c'])
        o.write('[%d|%s|%s] %s\n' % (i, p['style'], ','.join(sorted(flags)), line))
    o.write('\n==== COMMENTS ====\n')
    for k, c in comments.items():
        o.write('C%s [%s %s]: %s\n' % (k, c['author'], c['date'], c['text']))
