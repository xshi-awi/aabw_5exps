"""Re-highlight the spots that still need the authors' decision (2026-11-22)."""
import sys
p = sys.argv[1]
s = open(p).read()
disc = open(sys.argv[2]).read().strip()
targets = [
    disc,
    "The sea ice anomalies (Fig. 1i-l) follow the temperature changes. In MH the winter sea ice area changes little (+9\\%). In LIG it decreases by 18\\%, mainly in the Ross sector, where the winter ice edge retreats south by about 5° of latitude beneath the warm and salty surface anomaly. In LGM and MIS3 the winter sea ice area grows by a factor of 3.7 to 4.0, from 6.0 million km\\textsuperscript{2} in PI to 24.3 and 22.0 million km\\textsuperscript{2}, and the ice edge moves north by about 10° of latitude on average and by 14–15° in the Weddell sector.",
    "Three simulated changes work together to produce this ice cover. The winter air over the ocean between 55 and 70°S is 14 K colder in LGM and 12 K colder in MIS3 than in PI. The surface water between 60 and 75°S stays within 0.2 K of its freezing point in every sector, against 1.5–2.0 K above freezing in PI, so the upper ocean holds almost no heat that could melt the ice or stop new ice from forming. The glacial westerlies are also stronger (Fig. 1o,p), which favours the northward export of ice. These changes reinforce each other, because ice that covers the ocean also stops the ocean from warming the air above it.",
    "the freshwater contribution is the larger of the two wherever sea ice is present",
]
for t in targets:
    n = s.count(t)
    assert n == 1, (n, t[:80])
    s = s.replace(t, '\\hl{' + t + '}')
open(p, 'w').write(s)
print('highlights now', s.count('\\hl{'))
