#!/usr/bin/env python3
"""11-27 12:04 user button GM: R3 paragraph says the gyre interior 'has little ice' (true for the figure's
2090-2099 file, 18% JJA mean concentration, and for the paper's PI, 8%). Usage: edits_1127f_r3text.py <response_new.tex>"""
import sys
p = sys.argv[1]
s = open(p, encoding='utf-8').read()
old = r'especially in the Weddell sector, where the interior of the gyre is almost free of ice.'
new = r'especially in the Weddell sector, where the interior of the gyre has little ice.'
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)
assert '\t' not in s and '\r' not in s
open(p, 'w', encoding='utf-8').write(s)
print('ok', p)
