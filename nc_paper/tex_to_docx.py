#!/usr/bin/env python
"""
Convert the LaTeX response letter to .docx while preserving the three styles:

  reviewer comment    plain black
  our reply           blue bold
  quoted manuscript   blue bold italic, in quotation marks

pandoc's LaTeX reader silently drops \\textcolor, so we go through an HTML
intermediate, where colour survives as a real Word run property.
"""
import re
import subprocess
from pathlib import Path

SRC = Path('SUBMISSION/04_response_letter.tex')
HTML = Path('/tmp/claude-22370/-work-ba1066-a270064-cc-projects-aabw-5exps/'
            'a0433b42-6a8d-4bfc-8edf-280a200280f5/scratchpad/letter.html')
OUT = Path('SUBMISSION/05_response_letter.docx')

s = SRC.read_text()

# ---- cut preamble and end matter, keep the body
s = s.split(r'\maketitle', 1)[1]
s = s.split(r'\end{document}', 1)[0]


def find_group(text, start):
    """Return (content, index_after) for a brace group starting at text[start]=='{'."""
    assert text[start] == '{'
    depth = 0
    i = start
    while i < len(text):
        if text[i] == '{' and (i == 0 or text[i - 1] != '\\'):
            depth += 1
        elif text[i] == '}' and text[i - 1] != '\\':
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
        i += 1
    raise ValueError('unbalanced braces')


def strip_cmd(t, cmd):
    """Replace \\cmd{X} with X, repeatedly, honouring nesting."""
    pat = '\\' + cmd + '{'
    while pat in t:
        i = t.index(pat)
        inner, j = find_group(t, i + len(pat) - 1)
        t = t[:i] + inner + t[j:]
    return t


def inline(t):
    """LaTeX inline markup -> HTML, for text already known to be one style."""
    t = t.replace('\\ldots{}', '&hellip;').replace('\\ldots', '&hellip;')
    t = t.replace('\\%', '%').replace('\\&', '&amp;').replace('\\#', '#')
    t = t.replace('\\_', '_').replace('\\$', '$')
    t = t.replace('~', ' ')
    t = re.sub(r'\\emph\{([^{}]*)\}', r'<em>\1</em>', t)
    t = re.sub(r'\\textbf\{([^{}]*)\}', r'<strong>\1</strong>', t)
    t = re.sub(r'\\citep?\{[^{}]*\}', '', t)
    t = re.sub(r'\\url\{([^{}]*)\}', r'\1', t)
    t = re.sub(r'\\ref\{[^{}]*\}', '', t)
    t = re.sub(r'\\label\{[^{}]*\}', '', t)
    # maths and symbols
    t = t.replace('$^\\circ$', '&deg;').replace('$\\circ$', '&deg;')
    t = t.replace('$\\pm$', '&plusmn;').replace('\\pm', '&plusmn;')
    t = t.replace('$\\alpha$', '&alpha;').replace('\\alpha', '&alpha;')
    # md_to_tex writes the Greek letter and its subscript as two separate
    # maths groups, e.g. $\sigma$$_2$, so matching only \sigma_2 missed every
    # occurrence: the generic \\[a-zA-Z]+ strip below then deleted \sigma and
    # left a bare "2". Normalise the split form first.
    t = re.sub(r'\$\\(sigma|gamma|alpha|beta|delta|Delta)\$\$_\{?(\w+)\}?\$',
               lambda m: '&%s;<sub>%s</sub>' % (m.group(1), m.group(2)), t)
    t = re.sub(r'\$\\(sigma|gamma|alpha|beta|delta|Delta)_\{?(\w+)\}?\$',
               lambda m: '&%s;<sub>%s</sub>' % (m.group(1), m.group(2)), t)
    t = re.sub(r'\\(sigma|gamma|alpha|beta|delta|Delta)_\{?(\w+)\}?',
               lambda m: '&%s;<sub>%s</sub>' % (m.group(1), m.group(2)), t)
    # bare Greek letters with no subscript
    t = re.sub(r'\$\\(sigma|gamma|beta|delta|Delta)\$',
               lambda m: '&%s;' % m.group(1), t)
    t = t.replace('\\approx', '&asymp;').replace('\\ge', '&ge;').replace('\\le', '&le;')
    t = t.replace('\\times', '&times;')
    t = t.replace("\\'e", 'é').replace('\\v{c}', 'č').replace("\\'", '')
    t = re.sub(r'\$([^$]*)\$', r'\1', t)          # drop remaining math delimiters
    t = re.sub(r'\^\{?(-?\d+)\}?', r'<sup>\1</sup>', t)
    t = re.sub(r'_\{?(\w)\}?', r'<sub>\1</sub>', t)
    t = t.replace('``', '&ldquo;').replace("''", '&rdquo;')
    t = t.replace('--', '&ndash;')
    t = re.sub(r'\\[a-zA-Z]+', '', t)             # any stragglers
    t = t.replace('{', '').replace('}', '')
    return re.sub(r'\s+', ' ', t).strip()


BLUE = '#0000CC'
out = []

i = 0
n = len(s)
while i < n:
    # ---------------------------------------------------------- figure
    if s.startswith(r'\begin{center}', i) and 'includegraphics' in s[i:i + 400] \
            and 'tabular' not in s[i:i + 400]:
        # images are plain centred material followed by a \small caption
        # paragraph, because sn-jnl.cls prints a "Fig. N" label on \caption*
        j = s.index(r'\endgroup', i) + len(r'\endgroup')
        blk = s[i:j]
        m = re.search(r'\\includegraphics\[[^\]]*\]\{([^{}]*)\}', blk)
        cap = ''
        mc = re.search(r'\\begingroup\\small\\noindent ', blk)
        if mc:
            cap = blk[mc.end():].split('\\par')[0]
        # keep the "Figure Rx.y (...)" label bold, as it is in the PDF
        cap = re.sub(r'\\textbf\{(Figure [^{}]*)\}', r'@@B@@\1@@/B@@', cap, count=1)
        cap = inline(strip_cmd(strip_cmd(cap, 'textbf'), 'textit'))
        cap = cap.replace('@@B@@', '<strong>').replace('@@/B@@', '</strong>')
        cap = re.sub(r'^\s*', '', cap)
        if m:
            src = Path('SUBMISSION') / m.group(1)
            out.append(f'<p><img src="{src.resolve()}" style="width:100%" /></p>')
        if not cap:
            raise SystemExit('figure block with no caption: ' + blk[:80])
        out.append(f'<p><em>{cap}</em></p>')
        i = j
        continue

    # ---------------------------------------------------------- section
    m = re.match(r'\\(sub)?section\*?\{', s[i:])
    if m:
        st = i + m.end() - 1
        title, j = find_group(s, st)
        tag = 'h2' if m.group(1) else 'h1'
        out.append(f'<{tag}>{inline(title)}</{tag}>')
        i = j
        continue

    # ---------------------------------------------------------- blue block
    if s.startswith(r'\textcolor{blue}{', i):
        st = i + len(r'\textcolor{blue}')
        inner, j = find_group(s, st)
        italic = '\\textit{' in inner
        txt = strip_cmd(strip_cmd(inner, 'textbf'), 'textit')
        txt = inline(txt)
        if txt:
            SENT = '\u2060'
            if italic:
                out.append(f'<p><strong><em>{SENT}{txt}</em></strong></p>')
            else:
                out.append(f'<p><strong>{SENT}{txt}</strong></p>')
        i = j
        continue

    # ---------------------------------------------------------- table
    if s.startswith(r'\begin{tabular}', i):
        j = s.index(r'\end{tabular}', i) + len(r'\end{tabular}')
        blk = s[i:j]
        body = blk.split('}', 2)[2]          # drop \begin{tabular}{lccc}
        out.append('<table border="1" cellspacing="0" cellpadding="4">')
        for row in body.split(r'\\'):
            row = row.replace(r'\hline', '').strip()
            if not row or row == r'\end{tabular}':
                continue
            row = row.replace(r'\end{tabular}', '').strip()
            if not row:
                continue
            cells = [inline(c) for c in row.split('&')]
            out.append('<tr>' + ''.join('<td>%s</td>' % c for c in cells) + '</tr>')
        out.append('</table>')
        i = j
        continue

    # ---------------------------------------------------------- enumerate
    if s.startswith(r'\begin{enumerate}', i):
        j = s.index(r'\end{enumerate}', i) + len(r'\end{enumerate}')
        blk = s[i:j]
        items = [x for x in blk.split(r'\item')[1:]]
        out.append('<ol>')
        for it in items:
            it = it.replace(r'\end{enumerate}', '')
            out.append('<li>' + inline(strip_cmd(it, 'textbf')) + '</li>')
        out.append('</ol>')
        i = j
        continue

    # ---------------------------------------------------------- plain text
    # search from i+1: a \begin{center} that wraps a table is not consumed by
    # the figure branch above, so searching from i would find this very
    # position, give j == i, and spin here forever
    nxt = [x for x in
           [s.find(r'\textcolor{blue}{', i + 1), s.find(r'\begin{center}', i + 1),
            s.find(r'\begin{tabular}', i + 1),
            s.find(r'\section', i + 1), s.find(r'\subsection', i + 1),
            s.find(r'\begin{enumerate}', i + 1)] if x != -1]
    j = min(nxt) if nxt else n
    chunk = s[i:j]
    chunk = chunk.replace(r'\medskip', '').replace(r'\clearpage', '')
    chunk = strip_cmd(chunk, 'textbf')
    for para in re.split(r'\n\s*\n', chunk):
        p = inline(para)
        if p:
            out.append(f'<p>{p}</p>')
    i = j

html = ('<!DOCTYPE html><html><head><meta charset="utf-8"><title>'
        'Response to Reviewers</title></head><body>\n'
        + '\n'.join(out) + '\n</body></html>')
HTML.write_text(html)

# letter_reference.docx carries A4 page size, the margins measured from the
# compiled PDF (left 100pt, right 124pt, top 77pt, bottom 187pt) and a serif
# body font, so the .docx lays out like the .pdf rather than as default
# Calibri on Letter.
REF = Path('letter_reference.docx')
cmd = ['pandoc', str(HTML), '-o', str(OUT), '--from=html', '--to=docx']
if REF.exists():
    cmd.append('--reference-doc=%s' % REF)
subprocess.run(cmd, check=True)

# ------------------------------------------------------------------ colour
# pandoc 2.18 drops inline CSS colour, so inject it into the run properties.
# Every run that carries our blue marker character set is identified by the
# CustomStyle pandoc assigns; simpler and more robust: re-open the docx and add
# <w:color w:val="0000CC"/> to every run whose text we know is a reply. We mark
# replies in the HTML with a zero-width sentinel and strip it here.
import zipfile, shutil, re as _re

tmp = OUT.with_suffix('.tmp.docx')
SENT = '\u2060'          # word joiner, invisible
with zipfile.ZipFile(OUT) as zin, zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename == 'word/document.xml':
            xml = data.decode('utf-8')
            # Colour the WHOLE paragraph, not just the run holding the
            # sentinel. A subscript or superscript makes pandoc split the text
            # into several runs, and the sentinel sits only in the first one,
            # so colouring per run left everything after the first sigma_2 or
            # 10^11 black for the rest of the paragraph.
            def colour_run(run):
                run = run.replace(SENT, '')
                if '<w:color' in run:
                    return run
                if '<w:rPr>' in run:
                    return run.replace('<w:rPr>',
                                       '<w:rPr><w:color w:val="0000CC" />', 1)
                return run.replace('<w:r>',
                                   '<w:r><w:rPr><w:color w:val="0000CC" /></w:rPr>', 1)

            def fix_para(m):
                para = m.group(0)
                if SENT not in para:
                    return para
                return _re.sub(r'<w:r>.*?</w:r>',
                               lambda r: colour_run(r.group(0)), para, flags=_re.S)
            xml = _re.sub(r'<w:p>.*?</w:p>', fix_para, xml, flags=_re.S)
            # any sentinel outside a <w:p> (should not happen) must still go
            xml = xml.replace(SENT, '')

            # pandoc rebuilds sectPr from its own defaults and ignores the one
            # in the reference doc, so set the page up here: A4 with the text
            # block measured off the compiled PDF, in twentieths of a point.
            SECT = ('<w:pgSz w:w="11906" w:h="16838"/>'
                    '<w:pgMar w:top="1540" w:right="2480" w:bottom="3740"'
                    ' w:left="2000" w:header="708" w:footer="708" w:gutter="0"/>')
            # pandoc emits a self-closing <w:sectPr />, so match both forms
            if _re.search(r'<w:sectPr\s*/>', xml):
                xml = _re.sub(r'<w:sectPr\s*/>',
                              '<w:sectPr>%s</w:sectPr>' % SECT, xml)
            elif '<w:sectPr' in xml:
                xml = _re.sub(r'<w:sectPr[^>]*>.*?</w:sectPr>',
                              '<w:sectPr>%s</w:sectPr>' % SECT, xml, flags=_re.S)
            else:
                xml = xml.replace('</w:body>',
                                  '<w:sectPr>%s</w:sectPr></w:body>' % SECT)
            data = xml.encode('utf-8')
        zout.writestr(item, data)
shutil.move(tmp, OUT)

n = 0
with zipfile.ZipFile(OUT) as z:
    n = z.read('word/document.xml').decode('utf-8').count('w:color w:val="0000CC"')
print(f'wrote {OUT}  ({n} blue runs)')
