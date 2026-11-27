#!/usr/bin/env python3
"""Reply letter (revision/response_new.tex) -> Word, looking like the compiled PDF.

tex2docx (~/bin/tex2docx, skill tex2docx) does the conversion; it knows standard LaTeX but not this
letter's own macros and drops inline colour. So this script
  1. prep:  rewrites the letter into plain LaTeX (\\comment, \\response, change env, \\rfig, \\claudenote,
            \\reviewer expanded) and puts ASCII markers where the PDF changes colour or style,
  2. runs tex2docx on that file (same folder, so figures/../letter_figs/*.png resolve),
  3. post:  walks the runs of the docx in order and applies the marker states
            (blue #00468C for replies, blue italic 10 pt indented for manuscript quotes, magenta sans 9 pt
            for \\claudenote; captions stay black as in the PDF), then deletes the markers.
Usage: python3 -s letter_to_word.py <revision dir with response_new.tex> <out.docx>
Colours, sizes and fonts were read from the compiled PDF with PyMuPDF (11-27)."""
import os
import re
import shutil
import subprocess
import sys
import zipfile

from lxml import etree

SRC_DIR, OUT = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
SRC = os.path.join(SRC_DIR, 'response_new.tex')
PREP = os.path.join(SRC_DIR, 'response_word.tex')

BON, BOFF, QON, QOFF, MON, MOFF = ('QZBONQZ', 'QZBOFFQZ', 'QZQONQZ', 'QZQOFFQZ', 'QZMONQZ', 'QZMOFFQZ')
MARK = re.compile('(QZ(?:BON|BOFF|QON|QOFF|MON|MOFF)QZ)')
BLUE, MAGENTA = '00468C', 'EC008C'


# ------------------------------------------------------------------ 1. prep
def arg(s, i):
    """s[i] == '{': return (content, index after the closing brace). Braces escaped as \\{ \\} are skipped."""
    assert s[i] == '{', s[i:i + 40]
    depth, j = 0, i
    while True:
        c = s[j]
        if c == '\\':
            j += 2
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1


def strip_comments(s):
    out = []
    for line in s.split('\n'):
        m = re.search(r'(?<!\\)%', line)
        if m is None:
            out.append(line)
        elif line[:m.start()].strip() == '':
            continue                       # whole-line comment: drop the line
        else:
            out.append(line[:m.start()])
    return '\n'.join(out)


def esc_comment(body):
    """\\comment bodies are typeset with % # & _ ~ as other characters; make them ordinary LaTeX."""
    body = re.sub(r'(?<!\\)%', r'\\%', body)
    body = re.sub(r'(?<!\\)#', r'\\#', body)
    body = re.sub(r'(?<!\\)&', r'\\&', body)
    body = re.sub(r'(?<!\\)_', r'\\_', body)
    body = re.sub(r'(?<!\\)~', r'\\textasciitilde{}', body)
    return body


def prep():
    s = open(SRC, encoding='utf-8').read()
    head, body = s.split('\\begin{document}', 1)
    body, tail = body.rsplit('\\end{document}', 1)
    # preamble: drop the letter's own macros (expanded below), keep packages, colours, captions, numbering
    drop = ('\\newcommand{\\rfig}', '\\newcommand{\\comment}', '\\newcommand{\\commentbody}',
            '\\newcommand{\\response}', '\\newenvironment{change}', '\\DeclareRobustCommand{\\hl}',
            '\\DeclareRobustCommand{\\docxcomment}', '\\DeclareRobustCommand{\\claudenote}',
            '\\newcommand{\\reviewer}', '\\pretocmd')
    head = '\n'.join(l for l in head.split('\n') if not l.lstrip().startswith(drop)) + '\n\\date{}\n'
    # \comment bodies first (their % is literal), everything else after the comments are stripped
    out, i = [], 0
    while True:
        k = body.find('\\comment{', i)
        if k < 0:
            out.append(strip_comments(body[i:]))
            break
        out.append(strip_comments(body[i:k]))
        _, j = arg(body, k + len('\\comment'))
        txt, j = arg(body, j)
        out.append('\n\n' + BOFF + esc_comment(txt) + '\n\n')
        i = j
    b = ''.join(out)
    b = re.sub(r'\\\\\[\d+pt\]', r'\\\\', b)     # \\[4pt] in the title block -> \\
    for n in ('docxcomment', 'hl'):
        assert '\\%s{' % n not in b, n          # not used in the letter any more; add a rule if they return

    def sub_cmd(b, name, fn, nargs):
        while True:
            k = b.find('\\%s{' % name)
            if k < 0:
                return b
            j, args = k + len(name) + 1, []
            for _ in range(nargs):
                a, j = arg(b, j)
                args.append(a)
            b = b[:k] + fn(*args) + b[j:]

    b = sub_cmd(b, 'claudenote', lambda t: MON + '[Note: ' + t + ']' + MOFF, 1)
    b = sub_cmd(b, 'reviewer', lambda t: '\n\n\\section*{' + BOFF + t + '}\n\n', 1)
    b = sub_cmd(b, 'rfig', lambda w, f, cap: ('\n\n\\begin{figure}[h]\\centering\\includegraphics[width=%s\\linewidth]'
                                             '{figures/%s}\\caption{%s}\\end{figure}\n\n' % (w, f, cap)), 3)
    b = re.sub(r'\\response\b\s*', BON + r'\\textbf{Response.} ', b)
    b = re.sub(r'\\begin\{change\}\s*', '\n\n' + QON, b)
    b = re.sub(r'\s*\\end\{change\}', QOFF + '\n\n', b)
    for bad in ('\\comment{', '\\rfig{', '\\claudenote{', '\\response', 'change}'):
        assert bad not in b, bad
    open(PREP, 'w', encoding='utf-8').write(head + '\\begin{document}' + b + '\\end{document}' + tail)
    return b


# ------------------------------------------------------------------ 3. post
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS = {'w': W}


def q(t):
    return '{%s}%s' % (W, t)


def set_rpr(r, color=None, italic=None, size=None, sans=False):
    rpr = r.find(q('rPr'))
    if rpr is None:
        rpr = etree.SubElement(r, q('rPr'))
        r.remove(rpr)
        r.insert(0, rpr)

    def put(tag, attrs):
        e = rpr.find(q(tag))
        if e is None:
            e = etree.SubElement(rpr, q(tag))
        for a, v in attrs.items():
            e.set(q(a), v)
    if sans:
        put('rFonts', {'ascii': 'Arial', 'hAnsi': 'Arial', 'cs': 'Arial'})
    if italic:
        put('i', {})
        put('iCs', {})
    if color:
        put('color', {'val': color})
    if size:
        put('sz', {'val': str(size)})
        put('szCs', {'val': str(size)})
    # schema order of rPr children (Word is strict about it)
    order = ['rStyle', 'rFonts', 'b', 'bCs', 'i', 'iCs', 'caps', 'smallCaps', 'strike', 'dstrike', 'outline', 'shadow',
             'emboss', 'imprint', 'noProof', 'snapToGrid', 'vanish', 'webHidden', 'color', 'spacing', 'w', 'kern',
             'position', 'sz', 'szCs', 'highlight', 'u', 'effect', 'bdr', 'shd', 'fitText', 'vertAlign', 'rtl', 'cs',
             'em', 'lang', 'eastAsianLayout', 'specVanish', 'oMath']
    kids = sorted(rpr, key=lambda e: order.index(etree.QName(e).localname) if etree.QName(e).localname in order else 99)
    for e in kids:
        rpr.remove(e)
        rpr.append(e)


MR = '{http://schemas.openxmlformats.org/officeDocument/2006/math}r'


def math_color(mr, color):
    """OMML run: w:rPr sits after m:rPr."""
    rpr = mr.find(q('rPr'))
    if rpr is None:
        rpr = etree.Element(q('rPr'))
        pos = 1 if mr.find('{http://schemas.openxmlformats.org/officeDocument/2006/math}rPr') is not None else 0
        mr.insert(pos, rpr)
    c = rpr.find(q('color'))
    if c is None:
        c = etree.SubElement(rpr, q('color'))
    c.set(q('val'), color)


def split_run(r):
    """Split a w:r whose text holds markers into runs of plain text and marker-only runs."""
    t = r.find(q('t'))
    if t is None or t.text is None or not MARK.search(t.text):
        return [r]
    parts = [p for p in MARK.split(t.text) if p != '']
    new = []
    parent = r.getparent()
    idx = parent.index(r)
    for p in parts:
        rr = etree.fromstring(etree.tostring(r))
        for e in rr.findall(q('t')):
            rr.remove(e)
        tt = etree.SubElement(rr, q('t'))
        tt.text = p
        tt.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
        new.append(rr)
    parent.remove(r)
    for k, rr in enumerate(new):
        parent.insert(idx + k, rr)
    return new


def post(docx):
    tmp = docx + '.tmp'
    with zipfile.ZipFile(docx) as z:
        xml = z.read('word/document.xml')
        items = [(i, z.read(i.filename)) for i in z.infolist()]
    root = etree.fromstring(xml)
    body = root.find(q('body'))
    state = {'blue': False, 'quote': False, 'mag': False}
    stats = {'blue_runs': 0, 'quote_paras': 0, 'mag_runs': 0, 'markers': 0}
    for p in body.iter(q('p')):
        sty = p.find('w:pPr/w:pStyle', NS)
        sty = sty.get(q('val')) if sty is not None else ''
        is_caption = 'aption' in sty
        in_table = any(a.tag == q('tbl') for a in p.iterancestors())
        has_quote_text = False
        for r in list(p.iter(q('r'), MR)):
            if r.tag == MR:
                if (state['blue'] or state['quote']) and not is_caption and not in_table:
                    math_color(r, BLUE)
                continue
            for rr in split_run(r):
                t = rr.find(q('t'))
                txt = t.text if t is not None and t.text else ''
                m = MARK.fullmatch(txt)
                if m:
                    tag = m.group(1)[2:-2]
                    stats['markers'] += 1
                    state['blue'] = {'BON': True, 'BOFF': False}.get(tag, state['blue'])
                    state['quote'] = {'QON': True, 'QOFF': False}.get(tag, state['quote'])
                    state['mag'] = {'MON': True, 'MOFF': False}.get(tag, state['mag'])
                    rr.getparent().remove(rr)
                    continue
                if state['mag']:
                    set_rpr(rr, color=MAGENTA, size=18, sans=True)
                    stats['mag_runs'] += 1
                elif state['quote'] and not is_caption and not in_table:
                    set_rpr(rr, color=BLUE, italic=True, size=20)
                    has_quote_text = has_quote_text or bool(txt.strip()) or rr.find('.//m:oMath', {'m': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}) is not None
                elif state['blue'] and not is_caption and not in_table:
                    set_rpr(rr, color=BLUE)
                    stats['blue_runs'] += 1
        if has_quote_text:
            stats['quote_paras'] += 1
            ppr = p.find(q('pPr'))
            if ppr is None:
                ppr = etree.Element(q('pPr'))
                p.insert(0, ppr)
            ind = ppr.find(q('ind'))
            if ind is None:
                ind = etree.SubElement(ppr, q('ind'))
            ind.set(q('left'), '340')        # 1.2 em at 10 pt = 12 pt = 240 twips, plus the text block margin feel
            ind.set(q('right'), '340')
            ind.set(q('firstLine'), '0')
    # paragraphs left empty by marker removal
    for p in list(body.iter(q('p'))):
        if not ''.join(p.itertext()).strip() and p.find('.//' + q('drawing')) is None and p.find(
                './/{http://schemas.openxmlformats.org/officeDocument/2006/math}oMath') is None \
                and p.find('.//' + q('sectPr')) is None and p.getparent().tag == q('body'):
            p.getparent().remove(p)
    left = MARK.search(etree.tostring(root, encoding='unicode'))
    assert left is None, 'marker left in the document: %s' % left.group(0)
    xml = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
        for info, data in items:
            z.writestr(info, xml if info.filename == 'word/document.xml' else data)
    os.replace(tmp, docx)
    return stats


if __name__ == '__main__':
    prep()
    env = dict(os.environ)
    r = subprocess.run([os.path.expanduser('~/bin/tex2docx'), PREP, '-o', OUT, '--check'],
                       cwd=SRC_DIR, env=env, capture_output=True, text=True)
    sys.stdout.write(r.stdout[-3000:])
    sys.stderr.write(r.stderr[-3000:])
    assert r.returncode == 0, r.returncode
    print('post:', post(OUT))
