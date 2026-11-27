#!/usr/bin/env python3
"""11-27 afternoon pass on build/revised.tex: follow the user's TexPage edits of the letter quotes
(TexPage 7e48fc9..0c1d2cc, 15:23-16:19 +0800). Usage: edits_1127c_revised.py <revised.tex>
Every anchor must match exactly once, so the script refuses to run twice."""
import sys

p = sys.argv[1]
s = open(p, encoding='utf-8').read()


def rep(old, new):
    global s
    c = s.count(old)
    assert c == 1, (c, old[:90])
    s = s.replace(old, new)


# R1 comment 13 quote: "stands out" -> "has the most pronounced change"
rep(r'In the Last Interglacial the Ross sector stands out.',
    r'In the Last Interglacial the Ross sector has the most pronounced change.')

# R1 comment 4 quote: "Weaker glacial transformation" -> "Weaker glacial WMT"; WMT is then defined in the abstract
rep(r'Using water mass transformation analysis in AWI-ESM simulations',
    r'Using water mass transformation (WMT) analysis in AWI-ESM simulations')
rep(r'Weaker glacial transformation coincides with a collapse',
    r'Weaker glacial WMT coincides with a collapse')

# R2 Lhardy quote, shortened by the user ("我缩减了请同步"): the paragraph becomes her five sentences.
# The citation is also put on the first sentence, which states the PMIP result of Lhardy et al. (2021).
old_start = r'The ability of the model to maintain an extensive glacial winter ice cover, and thereby to suppress open-ocean convection, is not a general feature of paleoclimate simulations.'
old_end = r'Proxy-based reconstructions support the direction and the rough size of the simulated change, but their uncertainty is large.'
i = s.find(old_start)
j = s.find(old_end)
assert s.count(old_start) == 1 and s.count(old_end) == 1 and 0 <= i < j
assert '\n\n' not in s[i:j], 'the Lhardy paragraph should be one paragraph'
new = (r'Many PMIP models underestimate glacial Southern Ocean sea ice and its seasonality \cite{Lhardy2021}. '
       r'In our simulations the LGM sea ice extent is 29.8 million km$^2$ in winter and 11.1 million km$^2$ in summer. '
       r'Both lie within the proxy-based estimates of about $33\pm5$ and $10\pm3$ million km$^2$ \cite{Lhardy2021}. '
       r'We have not tested why the model does well here. '
       r'The relatively fine ocean mesh near Antarctica may help, and the simulated glacial climate is cold enough to keep the surface water around Antarctica at its freezing point.')
s = s[:i] + new + s[j + len(old_end):]

assert '\t' not in s and '\r' not in s
open(p, 'w', encoding='utf-8').write(s)
print('ok', p)
