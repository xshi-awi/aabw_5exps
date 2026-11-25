"""2026-11-25, user button FT: Figure R3 is now drawn as the sea ice of SI Fig. A6 (smoothed
1-degree colours) with the correct PI years 2300-2399; caption updated. Anchors match once."""
import sys
p = sys.argv[1]
s = open(p).read()
old = r"""New analysis. Southern Ocean sea ice concentration in the pre-industrial simulation (last 100 years) and in the OSI SAF satellite record for 1979–2008 (sea ice concentration climate data record, version 3.0). (a, b) September, the month with the most ice. (c) Model minus observations in September. (d, e) February, the month with the least ice. In (a, b, d, e) the concentration is drawn as sea ice is drawn in manuscript Fig. 2. Every value of 1\% or more is coloured, values between 1 and 10\% in the darkest blue, and only cells with less than 1\% ice are left white, so thin and sparse ice remains visible. The dashed orange lines (model) and solid black lines (observations) mark the 15\% ice edge, and the dotted lines in the same colours mark the 1\% line."""
new = r"""New analysis. Southern Ocean sea ice concentration in the pre-industrial simulation (model years 2300--2399) and in the OSI SAF satellite record for 1979--2008 (sea ice concentration climate data record, version 3.0). (a, b) September, the month with the most ice. (c) Model minus observations in September. (d, e) February, the month with the least ice. In (a, b, d, e) the colours are drawn in the same way as the sea ice in Supplementary Fig. A6. Model and observations are first averaged onto a 1° grid. Cells with less than 1\% ice are left out, and the field is then smoothed with a Gaussian filter 1.5 grid cells wide, which also fills those cells from their neighbours. Values below 10\% are shown in the darkest blue. The dashed orange lines (model) and solid black lines (observations) mark the 15\% ice edge of the unsmoothed fields."""
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)
open(p, 'w').write(s)
print('ok')
