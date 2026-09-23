#!/usr/bin/env python
"""
Generate the LaTeX response letter from RESPONSE_LETTER.md, so the markdown,
LaTeX and docx versions cannot drift apart.

Styling follows SUBMISSION/example_response.tex:
  reviewer comment (markdown blockquote with *italics*)  plain black
  our reply (plain markdown paragraph)                   blue bold
  quoted manuscript text (blockquote with > *"..."*)     blue bold italic
"""
import re
from pathlib import Path

SRC = Path('RESPONSE_LETTER.md')
OUT = Path('SUBMISSION/04_response_letter.tex')
HEAD = Path('tex_header.tex')

s = SRC.read_text()


def esc(t):
    t = t.replace('\\', '')
    for a, b in [('&', r'\&'), ('%', r'\%'), ('#', r'\#'), ('_', r'\_'),
                 ('{', r'\{'), ('}', r'\}'), ('~', r'\textasciitilde{}'),
                 ('^', r'\textasciicircum{}'), ('$', r'\$')]:
        t = t.replace(a, b)
    t = t.replace('“', '``').replace('”', "''").replace('’', "'").replace('‘', "'")
    t = t.replace('–', '--').replace('—', '---').replace('…', r'\ldots{}')
    t = t.replace('°', r'$^\circ$').replace('±', r'$\pm$').replace('×', r'$\times$')
    t = t.replace('≈', r'$\approx$').replace('≥', r'$\ge$').replace('≤', r'$\le$')
    # gamma_n is a single symbol: handle it before the bare gamma, or the
    # trailing _n is escaped to \_n and renders as a literal underscore
    t = t.replace('\u03b3\\_n', r'$\gamma_n$').replace('\u03b3\u2099', r'$\gamma_n$')
    t = t.replace('α', r'$\alpha$').replace('β', r'$\beta$').replace('γ', r'$\gamma$')
    t = t.replace('σ', r'$\sigma$').replace('Δ', r'$\Delta$')
    t = t.replace('é', r"\'e").replace('č', r'\v{c}')
    t = re.sub(r'\^\-([0-9])', r'$^{-\1}$', t)
    t = re.sub(r'([0-9])\^([0-9]+)', r'\1$^{\2}$', t)
    t = t.replace('¹¹', r'$^{11}$').replace('¹³', r'$^{13}$').replace('²', r'$^2$')
    t = t.replace('⁻³', r'$^{-3}$').replace('⁻¹', r'$^{-1}$').replace('⁻²', r'$^{-2}$')
    t = t.replace('₂', r'$_2$').replace('ₙ', r'$_n$')
    t = t.replace('−', '-').replace('⁻', '-').replace('‰', r'\textperthousand{}')
    t = t.replace('⁰', r'$^0$').replace('¹', r'$^1$').replace('³', r'$^3$')
    t = t.replace('⁴', r'$^4$').replace('⁵', r'$^5$').replace('→', r'$\to$')
    t = t.replace('εNd', r'$\varepsilon$Nd').replace('δ', r'$\delta$')
    t = ''.join(c if ord(c) < 128 else '' for c in t)
    return t


def inline(t):
    t = esc(t)
    t = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', t)
    t = re.sub(r'\*(.+?)\*', r'\\textit{\1}', t)
    return t


out = []
i = 0
lines = s.split('\n')
n = len(lines)
while i < n:
    ln = lines[i]

    # image
    m = re.match(r'!\[\]\(([^)]+)\)', ln.strip())
    if m:
        # sn-jnl.cls redefines \caption* so that it still prints a "Fig. N"
        # label, which we do not want next to our own "Figure R2.3" numbering.
        # The image and its caption are therefore set as ordinary centred
        # material rather than as a float with a caption.
        out.append('\n\\begin{center}')
        out.append('\\includegraphics[width=\\textwidth]{%s}' % m.group(1))
        out.append('\\end{center}')
        # caption is the next bold paragraph
        j = i + 1
        while j < n and not lines[j].strip():
            j += 1
        cap = []
        while j < n and lines[j].strip():
            cap.append(lines[j]); j += 1
        out.append('\\begingroup\\small\\noindent %s\\par\\endgroup\n'
                   % inline(' '.join(cap)))
        i = j
        continue

    # headings
    if ln.startswith('# '):
        out.append('\\clearpage\n\\section*{%s}' % inline(ln[2:]))
        i += 1; continue
    if ln.startswith('## '):
        out.append('\\subsection*{%s}' % inline(ln[3:]))
        i += 1; continue
    if ln.startswith('### '):
        out.append('\\subsubsection*{%s}' % inline(ln[4:]))
        i += 1; continue

    # blockquote block
    if ln.startswith('>'):
        # A blank line ends the quotation. Continuing across one would merge a
        # manuscript quotation with the reviewer comment that follows it, which
        # both fuses two different styles into one paragraph and loses the break
        # the reader needs between them.
        blk = []
        while i < n and lines[i].startswith('>'):
            blk.append(lines[i].lstrip('> ').rstrip())
            i += 1
        text = ' '.join(x for x in blk if x)
        # a quotation of manuscript text looks like  *"..."*
        if re.match(r'^\*".*"\*$', text.strip()):
            inner = text.strip()[2:-2]
            out.append('\\medskip\n\\noindent\\textcolor{blue}{\\textbf{\\textit{``%s\'\'}}}\n\\medskip'
                       % esc(inner))
        else:
            # A reviewer comment. The example letter sets these in plain black
            # roman with no emphasis, so the surrounding markdown italics that
            # mark the block as quoted material are dropped rather than carried
            # through to \textit.
            body = text.strip()
            if body.startswith('*') and body.endswith('*'):
                body = body[1:-1]
            out.append('\\medskip\n\\noindent %s\n\\medskip' % inline(body))
        continue

    # blank
    if not ln.strip():
        out.append('')
        i += 1; continue

    # ordinary paragraph = our reply, in blue bold
    para = []
    while i < n and lines[i].strip() and not lines[i].startswith(('#', '>', '![')):
        para.append(lines[i].rstrip()); i += 1
    text = ' '.join(para)
    if text.startswith('---'):
        out.append('')
        continue
    out.append('\\noindent\\textcolor{blue}{\\textbf{%s}}\n' % inline(text))

body = '\n'.join(out)
header = HEAD.read_text() if HEAD.exists() else ''
OUT.write_text(header + body + '\n\n\\end{document}\n')
print('wrote', OUT, len(body), 'chars')
