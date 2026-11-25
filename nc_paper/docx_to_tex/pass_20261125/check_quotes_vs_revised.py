"""Check every sentence of each \\begin{change} quote in the letter against revised.tex.
Usage: python3 check_quotes_vs_revised.py response_new.tex revised.tex revised.aux
\\ref is replaced by the number in the aux file and citations are dropped on both sides;
remaining MISSING lines are often notation only (±, sigma_2, degree signs) and need a look."""
import re, sys
L = open(sys.argv[1], encoding='utf8').read()
R = open(sys.argv[2], encoding='utf8').read()
aux = open(sys.argv[3], encoding='utf8').read()
labels = dict(re.findall(r'\\newlabel\{([^}]*)\}\{\{([^}]*)\}', aux))
R = re.sub(r'(?m)(?<!\\)%.*$', '', R)
R = re.sub(r'\\ref\{([^}]*)\}', lambda m: labels.get(m.group(1), '??'), R)
R = re.sub(r'\\citet\{[^}]*\}', 'CITET', R)
def norm(s):
    s = re.sub(r'\\cite\{[^}]*\}', '', s)
    s = re.sub(r'\((?:[^()]*?(?:et al\.|and [A-Z][a-z]+|OSI SAF|Marshall)[^()]*?\d{4}[a-z]?)\)', '', s)  # (Author, year; ...)
    s = re.sub(r'[A-Z][a-zé]+ et al\. \(\d{4}\)', 'CITET', s)
    s = s.replace('CITET et al.', 'CITET')
    s = re.sub(r'\\textsuperscript\{([^}]*)\}', r'^\1', s)
    s = re.sub(r'\$\^\{?([^$}]*)\}?\$', r'^\1', s)
    s = re.sub(r'\\hl\{', '', s)
    s = s.replace('``', '').replace("''", '').replace('~', ' ').replace('--', '-').replace('–', '-').replace('—', '-')
    s = re.sub(r"\\'e", 'é', s); s = re.sub(r'\\[a-zA-Z]+', ' ', s)
    s = re.sub(r'[{}$\\]', '', s)
    s = re.sub(r'\s+([,.;)])', r'\1', s); s = re.sub(r'\(\s+', '(', s)
    s = re.sub(r'\s+', ' ', s).strip().lower()
    return s
RN = norm(R)
for k, m in enumerate(re.finditer(r'\\begin\{change\}(.*?)\\end\{change\}', L, re.S)):
    q = norm(m.group(1)); line = L[:m.start()].count('\n') + 1
    sents = [x.strip() for x in re.split(r'(?<=[.!?])\s+(?=[a-z0-9(])', q) if len(x.strip()) > 25]
    miss = [x for x in sents if x.rstrip('.') not in RN]
    print(f'#{k+1} line {line}: {len(sents)-len(miss)}/{len(sents)} sentences found')
    for x in miss: print('    MISSING:', x[:160])
