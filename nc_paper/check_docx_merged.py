#!/usr/bin/env python
"""
Has every paragraph of the docx been merged into the markdown source?

Xiaoxu edits the response letter directly in Word, while the .tex and .docx are
regenerated from RESPONSE_LETTER.md. Anything he wrote that has not been carried
back into the markdown is destroyed by the next build. That happened once: a
regenerated docx overwrote his version on TexPage, and only a local copy saved
it.

Relying on remembering to merge is not good enough, so run this before any
rebuild or push. It reports paragraphs present in the docx but absent from the
markdown. Exit status 1 means something would be lost.

    python check_docx_merged.py                    # checks the default pair
    python check_docx_merged.py other.docx         # checks another docx

Paragraphs are compared on normalised text, ignoring whitespace, quote glyphs
and LaTeX-only markup, so cosmetic differences do not raise false alarms. The
figure captions and the reviewer comments are skipped, since the markdown holds
those in a different form.
"""
import re
import sys
import html
import zipfile
import unicodedata
from pathlib import Path

DOCX = Path('SUBMISSION/response_letter_R1_1013.docx')
MD = Path('RESPONSE_LETTER.md')
MIN_LEN = 60          # ignore headings and one-line fragments


def norm(t):
    """Normalise for comparison: markup, quote glyphs and spacing are noise."""
    t = unicodedata.normalize('NFKC', t)
    t = t.replace('’', "'").replace('‘', "'")
    t = t.replace('“', '"').replace('”', '"')
    t = t.replace('–', '-').replace('—', '-')
    t = re.sub(r'\*+', '', t)                 # markdown emphasis
    t = re.sub(r'^>\s*', '', t, flags=re.M)   # blockquote markers
    t = re.sub(r'!\[\]\([^)]*\)', '', t)      # images
    t = re.sub(r'[^0-9a-zA-Z]+', ' ', t)      # punctuation and symbols
    # Word flattens superscripts and subscripts into bare digits, so 10^11
    # reads as 1011 and sigma_2 as sigma2. Drop digits attached to letters and
    # collapse digit runs, otherwise identical text looks different.
    t = re.sub(r'\d+', '#', t)
    t = re.sub(r'[#\s]+', ' # ', t)          # one token per number run
    return ' '.join(t.split()).lower()


def docx_paragraphs(path):
    z = zipfile.ZipFile(path)
    x = z.read('word/document.xml').decode('utf-8')
    x = re.sub(r'<w:del [^>]*>.*?</w:del>', '', x, flags=re.S)   # accept revisions
    out = []
    for p in re.findall(r'<w:p [^>]*>.*?</w:p>|<w:p>.*?</w:p>', x, re.S):
        t = html.unescape(''.join(re.findall(r'<w:t[^>]*>(.*?)</w:t>', p, re.S)))
        if t.strip():
            out.append(t.strip())
    return out


def is_our_own_output(path):
    """True if this docx was produced by tex_to_docx, not edited by Xiaoxu.

    Our converter writes blue replies as runs coloured 0000CC and marks nothing
    else. A docx saved out of Word keeps that colour too, so colour alone does
    not separate them; what does is that our output never carries revision
    marks or highlighting, because we never add any.
    """
    x = zipfile.ZipFile(path).read('word/document.xml').decode('utf-8')
    edited = ('<w:ins ' in x) or ('<w:del ' in x) or ('w:highlight' in x)
    ours = 'w:color w:val="0000CC"' in x
    return ours and not edited


def formatting_report(path):
    """Report emphasis choices the text comparison cannot see.

    norm() strips markdown emphasis so that cosmetic differences do not raise
    false alarms, which means a change like removing the italics from a
    reviewer comment passes silently. Compare the styling of the docx against
    the convention the build applies: reviewer comments plain black, replies
    blue, quoted manuscript text blue italic.
    """
    x = zipfile.ZipFile(path).read('word/document.xml').decode('utf-8')
    x = re.sub(r'<w:del [^>]*>.*?</w:del>', '', x, flags=re.S)
    out = []
    for p in re.findall(r'<w:p [^>]*>.*?</w:p>|<w:p>.*?</w:p>', x, re.S):
        t = html.unescape(''.join(re.findall(r'<w:t[^>]*>(.*?)</w:t>', p, re.S))).strip()
        if len(t) < MIN_LEN or t.lstrip().startswith(('Figure R', 'Table R')):
            continue
        runs = [r for r in re.findall(r'<w:r[ >].*?</w:r>', p, re.S)
                if re.search(r'<w:t[^>]*>', r)]
        if not runs:
            continue
        ital = all(('<w:i/>' in r or '<w:i ' in r or '<w:i />' in r) for r in runs)
        blue = '0000CC' in p or '0000cc' in p
        quoted = t.lstrip().startswith(('"', '\u201c'))
        has_quote = ('\u201c' in t) or ('"' in t)
        if ital and not quoted and not has_quote:
            out.append('  italic but not a quotation: %s' % t[:90])
        if quoted and not ital:
            out.append('  manuscript quotation not italic: %s' % t[:90])
    if out:
        return ['formatting differs from the convention in %d place(s):' % len(out)] + out[:12]
    return []


def main():
    docx = Path(sys.argv[1]) if len(sys.argv) > 1 else DOCX
    if not docx.exists():
        print('no docx at %s, nothing to check' % docx)
        return 0
    if is_our_own_output(docx):
        print('%s carries no edits of its own (no revisions, no highlighting);'
              ' nothing to merge' % docx.name)
        return 0
    md_norm = norm(MD.read_text())
    missing = []
    for t in docx_paragraphs(docx):
        if len(t) < MIN_LEN:
            continue
        if t.lstrip().startswith(('Figure R', 'Table R')):
            continue                      # captions are rewritten by the build
        if 'Draft covering Reviewer 1 only' in t:
            continue                      # our own banner, added after the split
        n = norm(t)
        if not n:
            continue
        # a paragraph counts as merged if a long run of it survives in the md
        probe = n[:160] if len(n) > 160 else n
        if probe not in md_norm:
            missing.append(t)

    fmt = formatting_report(docx)

    if not missing:
        print('all %s paragraphs are in %s' % (docx.name, MD.name))
        if fmt:
            print()
            for line in fmt:
                print(line)
        return 0

    print('%d paragraph(s) in %s are NOT in %s.' % (len(missing), docx.name, MD.name))
    print('Rebuilding would destroy them. Merge them first.\n')
    for t in missing:
        print('  - %s' % (t[:200] + ('...' if len(t) > 200 else '')))
    return 1


if __name__ == '__main__':
    sys.exit(main())
