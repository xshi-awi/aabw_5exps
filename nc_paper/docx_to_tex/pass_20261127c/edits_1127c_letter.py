#!/usr/bin/env python3
"""11-27 afternoon pass on revision/response_new.tex (TexPage 0c1d2cc): the user's notes of
15:23-16:19 +0800. Quotes of the manuscript are cut from the edited revised.tex, with \\ref replaced
by the numbers in revised.aux and \\cite by author-year text.
Usage: python3 edits_1127c_letter.py <response_new.tex> <edited revised.tex> <revised.aux>
All LaTeX goes in raw strings. Every anchor must match exactly once."""
import re
import sys

path, rev_path, aux_path = sys.argv[1], sys.argv[2], sys.argv[3]
L = open(path, encoding='utf8').read()
R = open(rev_path, encoding='utf8').read()
AUX = dict(re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]*)\}', open(aux_path, encoding='utf8').read()))
T = '%claude回复（11-27 下午）：'

CITE = {
    'Pellichero2018': 'Pellichero et al., 2018', 'Bailey2023': 'Bailey et al., 2023',
    'Lhardy2021': 'Lhardy et al., 2021', 'Gong1999': 'Gong and Wang, 1999',
    'Thompson2000': 'Thompson and Wallace, 2000', 'Marshall2003': 'Marshall, 2003',
    'HallVisbeck2002': 'Hall and Visbeck, 2002', 'SenGupta2006': 'Sen Gupta and England, 2006',
    'Sallee2010': 'Sallée et al., 2010', 'Cerovecki2013': 'Cerovečki et al., 2013',
    'Abernathey2016': 'Abernathey et al., 2016',
}


def cites(keys):
    parts = []
    for k in keys.split(','):
        k = k.strip()
        assert k in CITE, f'no author-year text for cite key {k}'
        parts.append(CITE[k])
    return '(' + '; '.join(parts) + ')'


def quotify(s):
    # "Pellichero et al. \cite{Pellichero2018}" -> "Pellichero et al. (2018)"
    def named(m):
        k = m.group(2)
        assert k in CITE and CITE[k].startswith(m.group(1)), (m.group(0), k)
        return m.group(1) + ' (' + CITE[k].split(', ')[-1] + ')'
    s = re.sub(r'([A-Z][\w\']+ et al\.)\s*\\cite\{(\w+)\}', named, s)
    s = re.sub(r'\s*\\cite\{([^}]+)\}', lambda m: ' ' + cites(m.group(1)), s)

    def ref(m):
        assert m.group(1) in AUX, f'label {m.group(1)} not in aux'
        return AUX[m.group(1)]
    s = re.sub(r'\\ref\{([^}]+)\}', ref, s)
    assert '\\cite' not in s and '\\ref' not in s
    return s


def para(start):
    hit = [l for l in R.split('\n') if l.startswith(start)]
    assert len(hit) == 1, f'revised paragraph start found {len(hit)} times: {start[:70]!r}'
    return hit[0]


def sentences(start, first, last=None):
    """From the sentence starting with `first` to the end of the sentence starting with `last`
    (or to the end of the paragraph when last is None)."""
    p = para(start)
    i = p.find(first)
    assert i >= 0 and p.count(first) == 1, f'first sentence not unique: {first[:60]!r}'
    if last is None:
        return p[i:].rstrip()
    k = p.find(last, i)
    assert k >= 0, f'last sentence not found: {last[:60]!r}'
    j = p.find('. ', k)
    j = len(p) if j < 0 else j + 1
    return p[i:j].rstrip()


def change(text):
    return '\\begin{change}\n``' + quotify(text).strip() + "''\n\\end{change}"


def rep(old, new, count=1):
    global L
    n = L.count(old)
    assert n == count, f'anchor found {n} times, expected {count}: {old[:90]!r}'
    L = L.replace(old, new)


# ================================================================ R1, model evaluation: quote the Discussion
q1 = sentences('In this study, we analyse five simulated climate states', 'For the PI state, our model reproduces')
q2 = para('Two factors contribute, and they can be separated.')
q3 = para('The second factor is the density range')
rep(r'''%在文章中我们如何改的？原文呢
''', r'''In the revised manuscript, the comparison with Pellichero et al. (2018) is discussed in the Discussion, which now reads:

''' + change(q1) + '\n\n' + change(q2) + '\n\n' + change(q3) + r'''
%在文章中我们如何改的？原文呢
''' + T + r'''加了。稿里对应的是讨论里和 Pellichero 比的三段，原样引在上面（第一段开头那句模拟量级可比是投稿版就有的，从 The forcing balance, however, differs 起是这次新加的）：热盐比例不同的原因分成积分域和密度范围两条。和 ORAS5 比的 WMT 只在信里，稿件里只有 ORAS5 的 MLD（Supplementary Fig. A12）。第二段里的 42.7%、89--91% 和你在给审稿人 2 的回复里删掉的那段是同一内容，要是嫌这里太长，可以只留第一、三段，告诉我就行。
''')

# ================================================================ R1, sea ice figure: the other file (trial)
rep(r'''Elsewhere the model has too little ice especially in the Weddell sector and the interior of the gyre is almost free of ice. This is where the model convects''',
    r'''Elsewhere the model has too little ice, especially in the Weddell sector, where the interior of the gyre is almost free of ice. This is where the model convects''')
rep(r'''\rfig{0.9}{../letter_figs/figR21_seaice_obs.png}{New analysis. Winter (JJA) sea ice concentration in the pre-industrial simulation (model years 2300--2399) compared with the OSI SAF satellite record for 1979--2008 (sea ice concentration climate data record, version 3.0). (a) The pre-industrial simulation. (b) Model minus observations.\label{fR:seaice}}
%先替换成另一个海冰文件，我看看效果如何。
''', r'''\rfig{0.9}{../letter_figs/figR21_seaice_obs_pi9099.png}{New analysis. Winter (JJA) sea ice concentration in the pre-industrial simulation (model years 2300--2399) compared with the OSI SAF satellite record for 1979--2008 (sea ice concentration climate data record, version 3.0). (a) The pre-industrial simulation. (b) Model minus observations. \claudenote{Trial version for the authors. This figure uses pi/a\_ice\_reg.nc and pi/a\_ice\_clim.nc, the files behind Supplementary Fig. A2, which hold a 2090--2099 mean of another run. The model years in this caption belong to the previous version, figR21\_seaice\_obs.png.}\label{fR:seaice}}
%先替换成另一个海冰文件，我看看效果如何。
''' + T + r'''换好了，用的是 A2 读的那两个文件（1° 的 pi/a\_ice\_reg.nc 和网格上的 pi/a\_ice\_clim.nc），画法、色标、差值算法都没变，图注末尾加了一条品红色注，提醒这是试看版。看的时候请留意三件事。这两个文件是另一个 run 的 2090--2099 十年平均，来历查不到，所以图注里的 model years 2300--2399 对这版不成立。换文件后 Weddell 环流中间涂成了深蓝（30°W--30°E、60--70°S 的 JJA 平均密集度 18%，原来那版 8%，那版大部分格点不到 1% 所以留白），上面这段的「环流中间几乎无冰」就说得重了，用这版的话我建议改成 has little ice。Ross 和 Amundsen--Bellingshausen 扇区冰缘与卫星差约 1° 仍成立（两版都约 1.2°）。想换回原来那版，只要把上面 \rfig 里的文件名改回 figR21\_seaice\_obs.png，两版的字都已放大，都在 TexPage 上。
''')

# ================================================================ R1 comment 4: abstract quote now uses WMT
rep(r'''``Weaker glacial WMT coincides with a collapse of the simulated abyssal overturning cell from about 11 to 2 Sv and with older abyssal water. Ideal ages at 4000 m remain below 500 years during interglacials but exceed 750 years basin-wide during glacials, reaching 1500 years in the Pacific.''
\end{change}
''', r'''``Weaker glacial WMT coincides with a collapse of the simulated abyssal overturning cell from about 11 to 2 Sv and with older abyssal water. Ideal ages at 4000 m remain below 500 years during interglacials but exceed 750 years basin-wide during glacials, reaching 1500 years in the Pacific.''
\end{change}
''' + T + r'''修订稿摘要同步改成 WMT。摘要前面第一次出现 water mass transformation 的地方补了 (WMT)，摘要里用缩写要先定义。
''')
assert 'Weaker glacial WMT coincides' in R and 'water mass transformation (WMT) analysis' in R

# ================================================================ R1 comment 13: Ross quote, manuscript synced
assert 'In the Last Interglacial the Ross sector has the most pronounced change.' in R

# ================================================================ R1 comment 20: quote the sentences
rep(r'''\comment{R1}{20. Line 209: Please add the appropriate reference(s) here to support this statement.}
%原文呢
''', r'''\comment{R1}{20. Line 209: Please add the appropriate reference(s) here to support this statement.}
''')
q20 = sentences('Beyond the mean state, dense water formation also responds',
                'The Southern Annular Mode (SAM) is the leading mode', 'Surface buoyancy forcing in turn sets the rate')
rep(r'''The definition of the mode now cites Gong and Wang (1999), Thompson and Wallace (2000) and Marshall (2003).
''', r'''The definition of the mode now cites Gong and Wang (1999), Thompson and Wallace (2000) and Marshall (2003). The relevant sentences of the Introduction now read:

''' + change(q20) + r'''
%原文呢
''' + T + r'''加了。原稿 209 行那句话挪进引言 SAM 那段以后，就是这里引的第二到第五句，和第 5 条的引文同一段，一字不差。
''')

# ================================================================ R2 opening
rep(r'''\response Thanks for the positive and construction comments, please see our point by point response below.''',
    r'''\response Thanks for the positive and constructive comments. Please see our point-by-point response below.''')

# ================================================================ R2 major 1: grammar and the merged paragraph
rep(r'''our transformation shares the structure of theirs. As in Fig.~\ref{fR:wmtR2} a and b, buoyancy gain at the lighter classes, a zero crossing, and densification at the heavier classes.''',
    r'''our transformation shares the structure of theirs. As Fig.~\ref{fR:wmtR2}a and b show, both have buoyancy gain at the lighter classes, a zero crossing, and densification at the heavier classes.''')
rep(r'''the light classes below γ =27.1 and the dense classes above γ =27.67,''',
    r'''the light classes below γ = 27.1 and the dense classes above γ = 27.67,''')
rep(r'''Their stated uncertainties are the mixed layer detection method, the instrumental errors in pressure, temperature and conductivity, and the spatial sampling, which is sparsest under winter ice. Some difference is therefore expected before any model bias enters. Moreover, their estimate is based on Argo floats, ship-based profiles and instrumented seals,  the data are sparse under winter sea ice and carry their own uncertainties, and their densification estimate of 5±5 Sv carries an uncertainty as large as the value itself. Some discrepancy is therefore expected before any model bias is considered, and the agreement in shape should be read in that light.  %最后我加的这些，跟前面有些重复，你看怎么合并一下''',
    r'''Their estimate is based on Argo floats, ship-based profiles and instrumented seals. Its stated uncertainties come from the mixed layer detection method, the instrumental errors in pressure, temperature and conductivity, and the spatial sampling, which is sparsest under winter sea ice, and their densification estimate of 5±5 Sv carries an uncertainty as large as the value itself. Some discrepancy is therefore expected before any model bias is considered, and the agreement in shape should be read in that light. %最后我加的这些，跟前面有些重复，你看怎么合并一下
''' + T + r'''合并了。你加的几句（数据来源、冬季冰下数据少、5±5 Sv 的不确定性、Some discrepancy is therefore expected）和原来的「Their stated uncertainties … sparsest under winter ice」「Some difference is therefore expected」说的是一回事，现在并成三句：数据来源一句，不确定性一句，结论一句（用你写的那句）。这条回复顺手改了几处语法：开头 construction 改成 constructive 并拆成两句；「As in Fig. a and b, buoyancy gain …」缺动词，改成 As Fig. a and b show, both have …；γ =27.1 补了空格。''')

# ================================================================ R2 Lhardy: quote from the shortened manuscript paragraph
qL = para('Many PMIP models underestimate glacial Southern Ocean sea ice and its seasonality')
rep(r'''\begin{change}
``Many PMIP models underestimate glacial Southern Ocean sea ice and its seasonality. In our simulations the LGM sea ice extent is 29.8 million km\textsuperscript{2} in winter and 11.1 million km\textsuperscript{2} in summer. Both lie within the proxy-based estimates of about 33 ± 5 and 10 ± 3 million km\textsuperscript{2} (Lhardy et al., 2021). We have not tested why the model does well here. The relatively fine ocean mesh near Antarctica may help, and the simulated glacial climate is cold enough to keep the surface water around Antarctica at its freezing point. ''
\end{change} %我缩减了请同步
''', change(qL).replace(r'km$^2$', r'km\textsuperscript{2}').replace(r'$33\pm5$', '33 ± 5').replace(r'$10\pm3$', '10 ± 3') + r''' %我缩减了请同步
''' + T + r'''修订稿讨论那段已改成你这五句。原段后面的内容一并删了：Shi et al. 2025 那句（同一套模拟的海冰以前发表过）、Green et al. 2022 那两句（多模式和硅藻重建比，AWI-ESM 冬季冰偏少，所以冰期偏盐不太可能是冰太多造成的）和最后两句（结论依赖海冰扩张是否真实、重建不确定性大）。我在第一句后补了 (Lhardy et al., 2021)，「很多 PMIP 模式低估冰期海冰」正是这篇的结论，不写出处容易被问。Green 2022 现在稿里不再引，只在给审稿人 3 的回复里出现；它那条提醒对我们其实是有利的论据，要不要在稿里留一句你定。
''')

assert '\t' not in L and '\r' not in L
open(path, 'w', encoding='utf8').write(L)
print('ok', path)
