"""Chunk check docx -> PDF text, with the expected format changes neutralised.
Neutralised: page-number lines, 'Authors' response:' labels (printed as 'Response.'), reviewer headings,
caption prefixes 'Figure Rn (x).' (printed 'Figure Rm. X.'), letter-figure numbers, table pipes."""
import json, re, sys, unicodedata
d = json.load(open(sys.argv[1]))
raw = unicodedata.normalize('NFKC', open(sys.argv[2]).read())
raw = '\n'.join(l for l in raw.split('\n') if not re.fullmatch(r'\s*\d+\s*', l))
raw = re.sub(r'\[Comment \d+ \(.*?\]', ' ', raw, flags=re.S)
def norm(s):
    s = unicodedata.normalize('NFKC', s).lower()
    s = re.sub(r'(fig\.?|figure)\s*r\s*\d+', 'FIGR', s)
    return re.sub(r'[^a-z0-9FIGR]', '', s)
N = norm(raw)
LABEL = re.compile(r"^\s*Authors[’']\s*response:?\s*$")
bad_paras = []
for i, p in enumerate(d['paras']):
    t = ''.join(r['t'] for r in p['runs'] if not (r.get('s') or r.get('dele')))
    if not t.strip() or LABEL.match(t) or t.startswith('Response to comments by Reviewer'):
        continue
    t = re.sub(r'^Figure R\d+\s*\(([^)]*)\)\.?', lambda m: 'Figure R0. ' + m.group(1) + '.', t)
    if p['style'] == 'TABLEROW':
        cells = [c.strip() for c in t.split('|') if c.strip()]
        miss = [c for c in cells if norm(c) not in N]
    else:
        w = t.split()
        chunks = [' '.join(w[k:k + 6]) for k in range(0, len(w), 6)]
        miss = [c for c in chunks if norm(c) and norm(c) not in N]
    if miss:
        bad_paras.append(i)
        print(f'[{i}] missing:', ' || '.join(m[:90] for m in miss[:5]))
print('docx paragraphs checked with text:', sum(1 for p in d['paras'] if ''.join(r['t'] for r in p['runs'] if not r.get('s')).strip()))
print('paragraphs with a missing chunk:', len(bad_paras), bad_paras)
