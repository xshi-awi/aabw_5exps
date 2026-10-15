#!/usr/bin/env python
"""
Fourth merge batch: propagate Xiaoxu's rewritten G1 block from the R1-only
draft into RESPONSE_LETTER.md, the full letter that is actually submitted.

He edited the R1 draft, so his rewrite landed in RESPONSE_LETTER_R1.md. The
full letter still carried the superseded opening and the superseded reanalysis
argument, which would have gone to the reviewers. The two files share this
block verbatim, so it is lifted across rather than retyped.
"""
import sys
from pathlib import Path

R1 = Path('RESPONSE_LETTER_R1.md').read_text()
P = Path('RESPONSE_LETTER.md')
s = P.read_text()
edits = []


def block(src, start, end, label):
    """Pull the text from start through end, inclusive, out of src."""
    i = src.find(start)
    j = src.find(end)
    if i < 0 or j < 0:
        sys.exit('BLOCK %r: start %s end %s' % (label, i, j))
    return src[i:j + len(end)]


def sub(old, new, label):
    global s
    n = s.count(old)
    if n != 1:
        sys.exit('ANCHOR %r matched %d times, expected 1\n---\n%s\n---' % (label, n, old[:300]))
    s = s.replace(old, new)
    edits.append(label)


# the opening summary, from his first sentence through the end of the ORAS5 caveat
new_open = block(R1,
                 'A question is pointed out, namely the comparison',
                 'essentially unusable in the seasonally ice-covered region.',
                 'opening summary')
old_open = block(s,
                 'A problem is pointed out, namely the disagreement',
                 'and we have bounded the rest of our claims accordingly.',
                 'old opening summary')
sub(old_open, new_open, 'G1 opening summary synced from the R1 draft')

# the Pellichero provenance paragraphs
new_prov = block(R1,
                 'The reviewer is right that Figure A4 showed only model curves.',
                 'mean used elsewhere in our manuscript.',
                 'provenance')
old_prov = block(s,
                 'The reviewer is right that Figure A4 showed only model curves.',
                 'austral winter mean used elsewhere in the paper.',
                 'old provenance')
sub(old_prov, new_prov, 'G1 Pellichero provenance synced')

# figure, caption and the discussion that follows, through the ORAS5 expansion
new_body = block(R1,
                 '![](letter_figs/figR20_wmt_merged.png)',
                 'continue to treat their published result as the observational reference.',
                 'body')
old_body = block(s,
                 '![](letter_figs/figR16_pi_vs_pellichero.png)',
                 'We have therefore compared against their published result rather than recomputing one.',
                 'old body')
sub(old_body, new_body, 'G1 figure, caption and ORAS5 comparison synced')

P.write_text(s)
print('applied %d edits:' % len(edits))
for e in edits:
    print('  -', e)
