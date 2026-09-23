#!/usr/bin/env python
"""
Split the fused numbered comments in the Specific comments sections.

The submitted draft ran each reviewer comment and our reply together into a
single blue paragraph, as

    **7.** *Line 127: Figure numbering ...* --- The appendix figures have been ...

The example response letter sets the reviewer's words in plain black and only
our reply in blue, with a blank line between them. Reproduce that here by
splitting each pair at the em dash, emitting the comment as a blockquote (which
md_to_tex renders black) and the reply as an ordinary paragraph (blue).

Comment text is restored verbatim from comments.txt where a numbered entry can
be matched, so that the elisions in the drafted version do not survive. The
number stays with the comment, as it does in the example.
"""
import re
import sys
from pathlib import Path

NC = Path(__file__).parent
MD = NC / 'RESPONSE_LETTER.md'
COMMENTS = NC / 'comments.txt'


def verbatim_comments():
    """Map reviewer number -> the comment exactly as written, from comments.txt."""
    txt = COMMENTS.read_text()
    # R1's numbered list runs from "Specific comments :" to the R2 header
    block = txt.split('Specific comments :', 1)[1].split('Reviewer #2', 1)[0]
    out = {}
    cur = None
    buf = []
    for line in block.splitlines():
        m = re.match(r'^\s*(\d+)\.\s*(.*)$', line)
        if m:
            if cur is not None:
                out[cur] = ' '.join(' '.join(buf).split())
            cur = int(m.group(1))
            buf = [m.group(2)]
        elif cur is not None and line.strip():
            buf.append(line.strip())
    if cur is not None:
        out[cur] = ' '.join(' '.join(buf).split())
    # the source has a stray leading quote on #1 and drops some full stops
    for k, v in out.items():
        v = v.lstrip("' ").strip()
        if v and v[-1] not in '.?!"”':
            v += '.'
        out[k] = v
    return out


def main():
    src = MD.read_text()
    verb = verbatim_comments()
    paras = re.split(r'(\n\s*\n)', src)

    n_split = 0
    n_restored = 0
    for idx, p in enumerate(paras):
        t = p.strip()
        if not re.match(r'^\*\*\d+\.\*\*', t):
            continue
        body = ' '.join(t.split())
        m = re.match(r'^\*\*(\d+)\.\*\*\s+\*(.+?)\*\s+—\s+(.*)$', body)
        if not m:
            print('SKIP (unparsable): %s' % body[:90], file=sys.stderr)
            continue
        num, comment, reply = int(m.group(1)), m.group(2), m.group(3)

        # prefer the untouched wording from the reviewer's own file
        if num in verb and '...' not in verb[num] and '…' not in verb[num]:
            if comment != verb[num]:
                n_restored += 1
            comment = verb[num]

        paras[idx] = '> **%d.** %s\n\n%s' % (num, comment, reply)
        n_split += 1

    MD.write_text(''.join(paras))
    print('split %d comment/reply pairs, restored %d to verbatim wording'
          % (n_split, n_restored))


if __name__ == '__main__':
    main()
