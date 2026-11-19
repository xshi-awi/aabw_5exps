"""Write response_new.tex from the run-level dump of response-JL.docx (dump_docx.py).

Content follows the docx, format follows texpage-sync/response_letter (STYLE.md + example preamble).
Struck text is dropped, yellow highlights become \\hl{} except the characters lost in an
earlier conversion (restored below from build/revised.tex), Word comments become visible
red notes, and letter figures are numbered R1.. in order of appearance.

Usage: python3 -I build_response_new.py dump.json example_preamble.tex out.tex report.json
"""
import json
import re
import sys

dump, preamble_src, out_tex, report_out = sys.argv[1:5]
D = json.load(open(dump))
P, COMMENTS = D['paras'], D['comments']
REPORT = {'struck_spans': [], 'restored': [], 'highlights_kept': [], 'notes': [], 'refs': [],
          'caption_prefix': [], 'split_quotes': []}

BLUE = {'0000CC', '0070C0', '0000FF'}
LABEL_RE = re.compile(r"^\s*Authors[’']\s*response:?\s*$")
HEADINGS = {0: 'TITLE', 2: 'Reviewer 1', 191: 'Reviewer 2', 233: 'Reviewer 3', 359: 'Reviewer 4',
            363: 'Editorial requirements'}
TAG = lambda i: 'R1' if i < 191 else 'R2' if i < 233 else 'R3' if i < 359 else 'R4' if i < 363 else 'ED'
FORCE_RESPONSE = {54}                 # black text in a response: the co-author's instruction
TABLE = (29, range(30, 36))
FIGFILE = {'media/image1.png': 'figR20_wmt_merged.png', 'media/image2.png': 'figR17_mld_pi_vs_obs.png',
           'media/image3.png': 'fig_age_labels-1.png', 'media/image4.png': 'figR7_mld_glacial_zoom.png',
           'media/image5.png': 'figR9_ross_sector.png', 'media/image6.png': 'figS_common_axis-1.png',
           'media/image7.png': 'fig03_labels-1.png', 'media/image8.png': 'figR4_sam_push_pull.png',
           'media/image9.png': 'figR5_sam_wind_ice.png', 'media/image10.png': 'figR2_moc_5exps.png',
           'media/image11.png': 'figR3_age_global.png'}
LABELS = {12: 'fR:wmt', 51: 'fR:mld', 64: 'fR:agelab', 102: 'fR:mldzoom', 114: 'fR:ross',
          124: 'fR:commonaxis', 132: 'fR:fig3lab', 198: 'fR:wmtR2', 242: 'fR:mldR3', 245: 'fR:wmtR3ab',
          251: 'fR:wmtR3cf', 271: 'fR:pushpull', 278: 'fR:windice', 295: 'fR:moc', 298: 'fR:ageglobal'}
R1MAP = {1: 'fR:wmt', 2: 'fR:mld', 3: 'fR:agelab', 4: 'fR:mldzoom', 5: 'fR:ross', 6: 'fR:commonaxis',
         7: 'fR:fig3lab'}
REFMAP_PARA = {241: {1: 'fR:mldR3'}, 247: {2: 'fR:wmtR3ab'}, 254: {1: 'fR:wmtR3cf'},
               276: {5: 'fR:windice'}}
# source comments next to numbers that the 2026-11-08 corrections (D6, D8) changed in revised.tex
D6NOTE = {79: 'abstract now says "from about 11 to 2 Sv" (D6, 2026-11-08)',
          80: 'abstract now says "from about 11 to 2 Sv" (D6, 2026-11-08)',
          294: ('revised.tex now: computed from the simulated vertical velocity; -11.7/-11.3/-11.4 Sv to '
                '-2.3/-1.7 Sv, roughly 80-86%; upper cell 18.5 Sv in PI against 20.1 and 21.5 Sv '
                '(D6/D8, 2026-11-08)'),
          295: 'the figure file is the D6 version (geographic binning); the docx showed the earlier one'}
OLDFIG = {102: 'docx showed an earlier version of this figure', 132: 'docx showed an earlier version of this figure'}


EDITORIAL = range(364, 369)   # whole paragraph bold in the docx (old letter style); keep the lead-in bold only


def items_of(i):
    out = []
    for r in P[i]['runs']:
        if 'cstart' in r:
            out.append({'cs': r['cstart']})
        elif 'cend' in r:
            out.append({'ce': r['cend']})
        elif 'img' in r:
            out.append({'img': r['img']})
        elif 'fn' in r or 'sym' in r:
            raise SystemExit(f'unexpected footnote/symbol in paragraph {i}')
        else:
            for ch in r['t']:
                out.append({'ch': ch, 'i': bool(r.get('i')), 'b': bool(r.get('b')), 'va': r.get('va'),
                            'h': bool(r.get('h')), 's': bool(r.get('s') or r.get('dele') or r.get('deltext'))})
    if i in EDITORIAL:
        text = ''.join(x.get('ch', '') for x in out)
        lead = text.find('. ') + 1
        n = 0
        for x in out:
            if 'ch' in x:
                x['b'] = n < lead
                n += 1
    return out


def live(items):
    """(string of non-struck characters, list of item indices) for text searches."""
    s, idx = [], []
    for k, it in enumerate(items):
        if 'ch' in it and not it['s']:
            s.append(it['ch'])
            idx.append(k)
    return ''.join(s), idx


def splice(items, i, start, end, raw, highlight=False, why=''):
    """Replace the live characters [start, end) with one raw-LaTeX item."""
    s, idx = live(items)
    a, b = idx[start], idx[end - 1]
    old = s[start:end]
    items[a:b + 1] = [it for it in items[a:b + 1] if 'ch' not in it] + [{'raw': raw, 'h': highlight}]
    if why:
        REPORT['restored'].append(dict(para=i, old=old, new=raw, why=why))
    return items


def find(items, text, after=0):
    s, _ = live(items)
    k = s.find(text, after)
    if k < 0:
        raise SystemExit(f'anchor not found: {text!r}')
    return k


def restore(i, items):
    """Characters lost when the earlier letter was generated and then turned into Word."""
    if i == 37:   # '?' stands for the citation that the old letter dropped
        k = find(items, 'calculated following ?') + len('calculated following ')
        splice(items, i, k, k + 1, r'\textit{(Berger, 1977)}',
               why='citation \\cite{berger1977long} lost (old letter printed "following , and"); '
                   'written as tex2letter renders it')
    if i == 141:
        k = find(items, '-≤ft(/Cp)Qnet')
        splice(items, i, k, k + len('-≤ft(/Cp)Qnet'), r'$-\left(\alpha/C_p\right)Q_{net}$',
               why='formula mangled in the old letter ("-\\le ft(/C_p)Q_net"); from revised.tex')
        k = find(items, 'S Ffw')
        splice(items, i, k, k + len('S Ffw'), r'$\beta S F_{fw}$',
               why='beta and subscript lost (old letter printed "S F_fw"); from revised.tex')
        # the rest of that highlight ("the", " in equation 1 in Method") marked the broken formula
        s, idx = live(items)
        for word in ('that is the ', ' in equation 1 in Method'):
            k = s.find(word)
            for j in range(k, k + len(word)):
                items[idx[j]]['h'] = False
        REPORT['restored'].append(dict(para=i, old='highlight on "the" and "in equation 1 in Method"',
                                       new='not highlighted',
                                       why='part of the same highlight as the lost formula; words kept as in docx'))
    return items


def figrefs(i, items):
    tag = TAG(i)
    m = REFMAP_PARA.get(i) or (R1MAP if tag == 'R1' else {1: 'fR:wmtR2'} if tag == 'R2' else {})
    while True:
        s, _ = live(items)
        mm = re.search(r'(Fig\.|Figure)(\s*)R(\d+)(?!\d)(?!\.\d)', s)
        if not mm:
            break
        n = int(mm.group(3))
        if n not in m:
            raise SystemExit(f'unmapped figure reference R{n} in paragraph {i}: {s[mm.start()-40:mm.end()+20]!r}')
        h = items[live(items)[1][mm.start(3)]]['h']
        splice(items, i, mm.start(2), mm.end(3), '~\\ref{%s}' % m[n], highlight=h)
        REPORT['refs'].append(dict(para=i, docx='%s R%d' % (mm.group(1), n), label=m[n]))
    return items


ESC = {'\\': r'\textbackslash{}', '{': r'\{', '}': r'\}', '$': r'\$', '&': r'\&', '%': r'\%', '#': r'\#',
       '_': r'\_', '^': r'\^{}', '~': r'\textasciitilde{}', '\t': ' ', '\u00a0': '~'}


def esc(s):
    return ''.join(ESC.get(c, c) for c in s)


def note(cid, anchor):
    c = COMMENTS[cid]
    a = anchor if len(anchor) <= 70 else anchor[:67].rstrip() + '...'
    REPORT['notes'].append(dict(id=cid, author=c['author'], date=c['date'][:10], anchor=anchor, text=c['text']))
    return r'\docxcomment{%s}{%s, %s}{%s}{%s}' % (int(cid) + 1, esc(c['author']), c['date'][:10],
                                                  esc(a), esc(c['text']))


def render(items, i, mode):
    """mode: text, quote (inside change), caption, comment."""
    # straight double quotes typed in Word become LaTeX opening/closing quotes, alternating
    items = [dict(it) for it in items]
    n_q = 0
    for it in items:
        if it.get('ch') == '"' and not it['s']:
            it.pop('ch')
            it['raw'] = '``' if n_q % 2 == 0 else "''"
            n_q += 1
    # struck spans for the report
    run = []
    for it in items + [{'end': 1}]:
        if 'ch' in it and it['s']:
            run.append(it['ch'])
        elif run and not ('cs' in it or 'ce' in it):
            REPORT['struck_spans'].append(dict(para=i, text=''.join(run)))
            run = []
    out, hl = [], []

    def styled(seq):
        res, cur, key = [], [], None
        for it in seq:
            k = (it.get('i') and mode in ('text', 'comment'), it.get('b') and mode in ('text', 'comment'),
                 it.get('va')) if 'ch' in it else ('raw',)
            if k != key and cur:
                res.append((key, cur))
                cur = []
            key = k
            cur.append(it)
        if cur:
            res.append((key, cur))
        s = ''
        for key, seq2 in res:
            if key == ('raw',):
                s += ''.join(x['raw'] for x in seq2)
                continue
            t = esc(''.join(x['ch'] for x in seq2))
            ital, bold, va = key
            if va == 'superscript':
                t = r'\textsuperscript{%s}' % t
            elif va == 'subscript':
                t = r'\textsubscript{%s}' % t
            if bold:
                t = r'\textbf{%s}' % t
            if ital:
                t = r'\textit{%s}' % t
            s += t
        return s

    def flush():
        if hl:
            txt = styled(hl)
            out.append(r'\hl{%s}' % txt)
            REPORT['highlights_kept'].append(dict(para=i, text=''.join(x.get('ch', '') or x.get('raw', '') for x in hl)))
            hl.clear()

    plain = []
    for k, it in enumerate(items):
        if 'ce' in it:
            flush()
            if plain:
                out.append(styled(plain))
                plain = []
            out.append(note(it['ce'], ANCH[it['ce']]) + ' ')
            continue
        if 'cs' in it or 'img' in it:
            continue
        if 'ch' in it and it['s']:
            continue
        if it.get('h'):
            if plain:
                out.append(styled(plain))
                plain = []
            hl.append(it)
        else:
            flush()
            plain.append(it)
    flush()
    if plain:
        out.append(styled(plain))
    s = ''.join(out)
    return re.sub(r'[ ]{2,}', ' ', s).strip()


def kind(i):
    p = P[i]
    s, _ = live(items_of(i))
    if i in HEADINGS:
        return 'heading'
    if any('img' in r for r in p['runs']):
        return 'image'
    if i == TABLE[0] or i in TABLE[1]:
        return 'table'
    has_note = any('cend' in r for r in p['runs'])
    if not s.strip():
        return 'empty_note' if has_note else 'empty'
    if LABEL_RE.match(s):
        return 'label'
    cols = {r.get('c') for r in p['runs'] if r['t'].strip() and not r.get('s')}
    if i in FORCE_RESPONSE or cols & BLUE:
        return 'response'
    return 'comment'


def split_quote(i, items):
    """Split a response paragraph into lead, quoted manuscript text and tail."""
    s, idx = live(items)
    if i == 110:   # the opening quote mark was struck together with the lead-in
        a = s.find('The MH shows')
        b = s.rfind('”')
        return items[:idx[a]], items[idx[a]:idx[b]], items[idx[b] + 1:]
    a, b = s.find('“'), s.rfind('”')
    if a < 0 or b < 0 or b - a < 100:
        return None
    return items[:idx[a]], items[idx[a] + 1:idx[b]], items[idx[b] + 1:]


def caption(i):
    items = items_of(i)
    s, _ = live(items)
    m = re.match(r'Figure R\d+\s*(?:\(([^)]*)\))?\.?\s*', s)
    ident = m.group(1)
    new = (ident[0].upper() + ident[1:] + '. ') if ident else ''
    h = items[live(items)[1][0]]['h']
    splice(items, i, 0, m.end(), esc(new), highlight=h)
    REPORT['caption_prefix'].append(dict(para=i, old=s[:m.end()], new=new))
    return render(figrefs(i, items), i, 'caption')


# text each Word comment is attached to (ranges may cross paragraphs)
ANCH, _open, _seg = {}, set(), {}
for _i in range(len(P)):
    for it in items_of(_i):
        if 'cs' in it:
            _open.add(it['cs'])
            _seg[it['cs']] = []
        elif 'ce' in it:
            seg = _seg.pop(it['ce'])
            _open.discard(it['ce'])
            t = ''.join(x['ch'] for x in seg if not x['s']).strip()
            ANCH[it['ce']] = t if t else ''.join(x['ch'] for x in seg).strip() + ' [struck text]'
        elif 'ch' in it:
            for c in _open:
                _seg[c].append(it)

body = []
state = 'none'
pending = False
consumed = set()
fig_no = 0


def emit_response_text(txt):
    global pending, state
    if state != 'response' or pending:
        body.append(r'\response ' + txt)
    else:
        body.append(txt)
    pending = False
    state = 'response'


for i in range(len(P)):
    if i in consumed:
        continue
    k = kind(i)
    if k == 'empty':
        st = ''.join(it['ch'] for it in items_of(i) if 'ch' in it and it['s'])
        if st.strip():
            REPORT['struck_spans'].append(dict(para=i, text=st))
        continue
    if k == 'heading':
        name = HEADINGS[i]
        s, _ = live(items_of(i))
        if name == 'TITLE':
            continue
        body.append('%' + '=' * 69)
        body.append('%% docx heading: %s' % s.strip())
        body.append(r'\reviewer{%s}' % name)
        state, pending = 'none', False
        continue
    if k == 'comment':
        body.append(r'\comment{%s}{%s}' % (TAG(i), render(figrefs(i, items_of(i)), i, 'comment')))
        state, pending = 'comment', False
        continue
    if k == 'label':
        pending, state = True, 'response'
        continue
    if k == 'empty_note':
        emit_response_text(render(items_of(i), i, 'text'))
        continue
    if k == 'image':
        img = [r['img'] for r in P[i]['runs'] if 'img' in r][0]
        j = i + 1
        while kind(j) == 'empty':
            j += 1
        if not live(items_of(j))[0].startswith('Figure R'):
            raise SystemExit(f'no caption after image in paragraph {i}')
        consumed.update(range(i + 1, j + 1))
        fig_no += 1
        if i in D6NOTE:
            body.append('% NOTE: ' + D6NOTE[i])
        if i in OLDFIG:
            body.append('% NOTE: ' + OLDFIG[i])
        body.append(r'\rfig{0.9}{../letter_figs/%s}{%s\label{%s}}' % (FIGFILE[img], caption(j), LABELS[i]))
        continue
    if k == 'table':
        if i == TABLE[0]:
            items = items_of(i)
            s, _ = live(items)
            m = re.match(r'Table R\d+\.\s*', s)
            splice(items, i, 0, m.end(), '')
            rows = []
            for r in TABLE[1]:
                cells, cur = [], []
                for run in P[r]['runs']:
                    if run['t'] == ' | ':
                        cells.append(cur)
                        cur = []
                    else:
                        cur.append(run)
                P_cells = []
                for c in cells:
                    P.append(dict(style='CELL', runs=c))
                    P_cells.append(render(items_of(len(P) - 1), r, 'text'))
                rows.append(P_cells)
            ncol = len(rows[0])
            tab = [r'\begin{table}[h!]', r'\caption{%s}' % render(items, i, 'caption'), r'\label{tR:bc}',
                   r'\centering\footnotesize\setlength{\tabcolsep}{3.5pt}',
                   r'\begin{tabular}{l%s}' % ('c' * (ncol - 1)), r'\toprule',
                   ' & '.join(rows[0]) + r' \\', r'\midrule']
            tab += [' & '.join(r) + r' \\' for r in rows[1:]]
            tab += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
            if state != 'response':
                body.append(r'\response')
                state = 'response'
            body.append('\n'.join(tab))
        continue
    # response paragraph
    items = restore(i, figrefs(i, items_of(i)))
    sp = split_quote(i, items)
    if i in D6NOTE:
        body.append('% NOTE: ' + D6NOTE[i])
    if sp is None:
        emit_response_text(render(items, i, 'text'))
        continue
    lead, quote, tail = sp
    REPORT['split_quotes'].append(i)
    lead_t = render(lead, i, 'text')
    if lead_t:
        emit_response_text(lead_t)
    elif state != 'response' or pending:
        emit_response_text('')
    body.append('\\begin{change}\n``%s\'\'\n\\end{change}' % render(quote, i, 'quote'))
    tail_t = render(tail, i, 'text')
    if tail_t:
        body.append(tail_t)

pre = open(preamble_src).read()
pre = pre[pre.index(r'\documentclass'):pre.index(r'\begin{document}')]
extra = r"""% ---- additions for this letter (aabw_5exps, Nature Communications revision) ----
\DeclareUnicodeCharacter{03B3}{\ensuremath{\gamma}}
% Yellow highlight kept from the co-authors' Word file. ulem (in every TeX Live) instead of soul,
% because soul before v3.0 breaks on UTF-8 characters under pdfLaTeX.
\usepackage[normalem]{ulem}
\DeclareRobustCommand{\hl}[1]{\bgroup\markoverwith{\textcolor{yellow}{\rule[-.5ex]{2pt}{2.6ex}}}\ULon{#1}}
% Word comment from the co-authors' file: number, author and date, the text it was attached to, the comment.
\DeclareRobustCommand{\docxcomment}[4]{{\upshape\sffamily\footnotesize\color{red}[Comment #1 (#2) on ``#3'': #4]}}
"""
head = r"""\begin{document}

\begin{center}
{\Large\bfseries Summary of Responses to Reviewers}\\[4pt]
{\large Glacial--interglacial regime shift in Southern Ocean dense water formation}\\[2pt]
Xiaoxu Shi, Jiping Liu, Chaoyuan Yang, Gerrit Lohmann
\end{center}

\bigskip
"""
top = ('% Response letter written from response-JL.docx (co-authors\' version, 2026-11-19).\n'
       '% Content follows the docx; layout follows the texpage-sync response-letter style.\n'
       '% Compile with pdfLaTeX, twice. Figures come from ../letter_figs via the figures/ folder.\n')
pre = top + pre.split('\n', 2)[2] if pre.startswith('%') else top + pre
with open(out_tex, 'w') as f:
    f.write(pre + extra + '\n' + head + '\n\n'.join(body) + '\n\n\\end{document}\n')
json.dump(REPORT, open(report_out, 'w'), ensure_ascii=False, indent=1)
print('figures', fig_no, 'notes', len(REPORT['notes']), 'struck spans', len(REPORT['struck_spans']),
      'highlights kept', len(REPORT['highlights_kept']), 'restored', len(REPORT['restored']),
      'refs', len(REPORT['refs']), 'quotes split', len(REPORT['split_quotes']))
