#!/usr/bin/env python3
"""11-27 05:48 user button GH: letter Fig. R3 (figR21_seaice_obs) without both 15% lines.
Caption sentence on the lines removed, claude note corrected. Usage: edits_1127b_letter.py <response_new.tex>"""
import sys
p = sys.argv[1]
s = open(p, encoding='utf-8').read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


rep(r'(b) Model minus observations. The solid black and dashed orange lines mark the observed and the simulated 15\% ice edge.\label{fR:seaice}}',
    r'(b) Model minus observations.\label{fR:seaice}}')
rep(r'(a) PI 海冰，画法和补充材料 A2 的海冰行一样（同一色标、同样平滑），叠了卫星和模式的 15% 冰缘线；(b) PI 减卫星。',
    r'(a) PI 海冰，画法和补充材料 A2 的海冰行一样（同一色标、同样平滑）；(b) PI 减卫星。两条 15% 冰缘线按你 05:48 的按钮 GH 都去掉了，文字里的冰缘数字是从数据算的，留着。')
assert '\t' not in s and '\r' not in s
open(p, 'w', encoding='utf-8').write(s)
print('ok', p)
