#!/usr/bin/env python3
"""11-27 11:53 user (via questions-ef): R3 keeps figR21_seaice_obs_pi9099.png; caption without the magenta
trial note and without the model years. Usage: edits_1127e_r3.py <response_new.tex>"""
import sys
p = sys.argv[1]
s = open(p, encoding='utf-8').read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


rep(r'sea ice concentration in the pre-industrial simulation (model years 2300--2399) compared with the OSI SAF',
    r'sea ice concentration in the pre-industrial simulation compared with the OSI SAF')
rep(r''' \claudenote{Trial version for the authors. This figure uses pi/a\_ice\_reg.nc and pi/a\_ice\_clim.nc, the files behind Supplementary Fig. A2, which hold a 2090--2099 mean of another run. The model years in this caption belong to the previous version, figR21\_seaice\_obs.png.}\label{fR:seaice}}''',
    r'''\label{fR:seaice}}''')
body = s.split('\\begin{document}', 1)[1]
assert '\\claudenote{' not in body
assert '\t' not in s and '\r' not in s
open(p, 'w', encoding='utf-8').write(s)
print('ok', p)
