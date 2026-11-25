"""2026-11-25 pass on revision/response_new.tex after the user's online edits (TexPage up to abeb2b3).

Only fixes the compile error and two obvious typos in her new caption, answers her own %-notes
in %claude回复 lines, adds the manuscript quote she asked for under comment 13, and puts the
checks of her new text into \\claudenote. Her wording is not changed. Every anchor must match
exactly once, so the script can be re-run on a freshly pulled version.
Usage: python3 edits_1125.py <response_new.tex> <quote_l114.txt>
"""
import sys

p, qfile = sys.argv[1], sys.argv[2]
s = open(p).read()
quote = open(qfile).read().strip()


def rep(old, new):
    global s
    n = s.count(old)
    assert n == 1, (n, old[:120])
    s = s.replace(old, new)


# ---- R2, LGM sea ice extent: her placeholder is filled first, while it is the only %claude回复 line
rep("""%这里你先算一下咱们lgm里面南极海冰的jas/jfm sie是多少？给我几个数在接下来的回复中
%claude回复：
""", """%这里你先算一下咱们lgm里面南极海冰的jas/jfm sie是多少？给我几个数在接下来的回复中
%claude回复：南半球海冰范围（SIE，密集度 >= 15% 的面积；模式取 100 年逐月平均，原生网格），单位百万 km2。LGM：JAS 平均 29.8，JFM 平均 11.1（9 月最大 30.4，2 月最小 8.9）。MIS3：27.6 和 9.1。PI：9.1 和 1.2。卫星 OSI SAF 1979-2008：18.1 和 4.5。LGM 冬季约为 PI 的 3.3 倍，夏季约 9 倍。数在 fig_seaice_obs/seaice_numbers.txt 第 4 节。要和重建值比（如 Gersonde et al. 2005），需另查文献核数。
""")

# ---- compile error and typos in her new caption of the ORAS5 figure
rep("""{（New analysis) Annual mean WMT""", """{(New analysis) Annual mean WMT""")
rep("""the seasonal sea ice zone(c)  and (c,d) south of 60°S""", """the seasonal sea ice zone and (c,d) south of 60°S""")
rep("""%这里图里面的图标要改成abcd，
""", """%这里图里面的图标要改成abcd，
%claude回复：图里四格已改成 (a)-(d)，旧图存 figures/versions/figR20b_wmt_oras5_v20261122.*。
\\claudenote{Panels (c,d) are winter (JJA) means south of 60°S, not annual means. Suggested caption: ``New analysis. WMT from our pre-industrial run and from ORAS5 (1999--2018), both with the sea ice contribution removed, for (a,b) the seasonal sea ice zone in the annual mean and (c,d) the region south of 60°S in JJA.''}
""")

# ---- R1 comment 4: the quoted passage is not in the manuscript
rep("""reaching 1500 years in the Pacific.''
\\end{change}
""", """reaching 1500 years in the Pacific.''
\\end{change}

\\claudenote{This quoted Discussion passage, starting ``The weaker WMT in the glacial simulations'', is not in revised.tex, neither in the new wording nor in the old one. Please add it to the Discussion, or quote the passage that is there.}
""")

# ---- R1 comment 9
rep("""``As a result, the MLD is strongly reduced and becomes confined to narrow coastal bands and polynyas.''
\\end{change}
""", """``As a result, the MLD is strongly reduced and becomes confined to narrow coastal bands and polynyas.''
\\end{change}

\\claudenote{Language: ``We have rephrased it as follows:'' reads better than ``We now rephrased it to:''. The quote does not match revised.tex line 111, which reads ``As a result, the deep mixed layers characteristic of open-ocean convection are strongly reduced and become confined to narrow coastal bands and polynyas'', followed by a figure reference and the area numbers. Please quote the sentence as it stands, or shorten the manuscript in the same way. The numbers removed here are still in the manuscript and in the reply to comment 10.}
""")

# ---- R1 comment 11
rep("""\\claudenote{revised.tex does not describe Fig. 1i-l yet. now modify it like above}
""", """\\claudenote{revised.tex does not describe Fig. 1i-l yet. now modify it like above}
\\claudenote{Checked before putting it into revised.tex, so it is not added yet. The new second sentence depends on the PI sea ice file. Against the overwritten PI file that Fig. 1i-l still uses, winter (JJA) ice in the western Weddell Sea (60--30°W, south of 60°S) shrinks in MH ($-$0.50 million km$^2$) and LIG ($-$0.41). Against the correct PI (model years 2300--2399) it grows in both ($+$0.04 and $+$0.14), and the ice of the whole Weddell sector grows too. Once Fig. 1 is redrawn with the correct file, this sentence would have to say the opposite. The LIG ice loss in the Ross sector holds with either file. Language: ``obviously'' is vague, and ``In LIG it also decreases strongly in the Ross sector'' says the same. Please write 10--15° with an en dash.}
""")

# ---- R1 comment 13: the manuscript passage she asked for, verbatim from revised.tex line 114
rep("""%这里加： begin change 写我们在文章中添加了哪些相关内容
""", """%这里加： begin change 写我们在文章中添加了哪些相关内容
%claude回复：已加在下面，引文从 revised.tex 第 114 行原样摘出。

In the revised manuscript this now reads:

\\begin{change}
``""" + quote + """''
\\end{change}

\\claudenote{Data check for the new reply: in LIG the Ross sector surface (60--75°S, JJA) is 0.97 K warmer and 0.25 psu saltier than in PI. The extra salt lowers the freezing point by only 0.015 K, so it is the warming, not the salinity, that puts the water 0.98 K further from freezing. Suggested: ``In LIG the surface of the Ross sector is about 1 K warmer than in PI and sits well above its freezing point.'' Language: ``Lines 146--148 describe the glacial experiments, which show stronger wind stress and more sea ice in the Ross sector. We therefore think the reviewer refers to the LIG experiment, which has stronger winds but less sea ice there.'' The reply now says twice that this is the reviewer's mechanism, and once is enough. The quoted manuscript passage still carries numbers from the overwritten PI sea ice file. With the correct file the Ross ice loss in LIG is 0.20 against a gain of 0.08 in the Weddell sector, so ``an order of magnitude more'' no longer holds, and the glacial gains earlier in the same paragraph are 0.61 (Ross) and 0.80 (Weddell) rather than 0.62 and 0.70.}
""")

# ---- R1 comment 17: unit under the colour bar of Fig. 3
rep("""\\label{fR:fig3lab}} %colorbar下面的单位怎么那么长
""", """\\label{fR:fig3lab}} %colorbar下面的单位怎么那么长
%claude回复：这是 EE2 定的单位 kg m-3 month-1 per metre of water column，意思是表层 1 m 水柱一个月的密度变化。可以换成数值完全相同的 kg m-2 month-1（穿过海面的密度通量），色标下只写这一个单位，图注里用一句话解释。要改的话 Fig. 3、4 和信里的 R8、R10 及图注一起改，等你点头。
""")

# ---- R1 comment 20: SAM references
rep("""%%在L209附近是这样的：
""", """%%在L209附近是这样的：
%claude回复：候选引用，凭记忆列出，加进 ref.bib 前要逐条核对原文。SAM 概念与指数：Gong & Wang 1999（GRL 26, 459-462，最早定义 AAO 指数）；Thompson & Wallace 2000（J. Climate 13, 1000-1016，annular modes，引用最高）；Marshall 2003（J. Climate 16, 4134-4143，站点 SAM 指数，观测研究最常用）。SAM 影响 WMT 的文章还没查，要做文献检索后再给。你点头后我核对文献、补 ref.bib、改 revised.tex 这两句，Hall & Visbeck (2002) 一起加。
""")
rep("""(line 142).} %按照我前面说的去引用
""", """(line 142).} %按照我前面说的去引用
%claude回复：好，和上面的 SAM 引用一起做（要改 revised.tex 和 ref.bib），等确认。
""")

# ---- R1 comments 21 and 22
rep("""请处理。
\\response \\docxcomment{6}""", """请处理。
%claude回复：明白。要改 revised.tex 2.4 节，给每处描述 Figure A5 的地方加子图字母。等确认后和第 22 条一起做。
\\response \\docxcomment{6}""")
rep("""as extensive sea ice coverage limits direct atmosphere-ocean exchange there.
""", """as extensive sea ice coverage limits direct atmosphere-ocean exchange there.
%claude回复：东南极沿岸从西往东确有 Cosmonaut Sea、Cooperation Sea、Davis Sea、Mawson Sea、Dumont d'Urville Sea 等名字，具体经度范围待核。哪几个海区该写进去，要先从 SAM 热通量合成里量出失热增加超过 15 W m-2 的范围，再对应海名；冰期那句的 south of ?°S 也从同一份数据量。原稿的 >15 W m-2 是当初看图读的，没算过（composite_sam/CLAUDE.md），这次一起从数据算。这是改 revised.tex 的活，等确认后做。
""")
rep("""{Rewrite!} %按照我前面说的改
""", """{Rewrite!} %按照我前面说的改
%claude回复：同上，和第 21 条一起改。
""")

# ---- R2, polynya circulation
rep("""%你确定有polynya circulation这个东西？
""", """%你确定有polynya circulation这个东西？
%claude回复：这不是标准术语，说法含糊。文献讲 polynya 时说的是风把冰推离岸、开阔水里不断结冰析盐，以及 polynya 里的小尺度对流和混合。原句想说的是：析出多少盐取决于模式里风把冰推开的程度，而不取决于 polynya 内部那些网格分辨不了的小尺度过程。建议信里改成 not on the small-scale processes inside each polynya, which the mesh is too coarse to resolve；修订稿第 271 行 rather than on a resolved polynya circulation 同样改成 rather than on small-scale processes inside the polynyas, which the mesh does not resolve。
""")

# ---- R2, Figs. 5 and 6 on one density scale
rep("""5和6都是3行5列的图。
""", """5和6都是3行5列的图。
%claude回复：对，审稿人要的是两种强迫放在同一个当量尺度上。现在信里的 Figure R10 只画了热和淡水的总量（2 行 x 5 列，已是同一密度通量单位）。把 A3（总热、LW+SW、SH+LH）和 A4（总淡水、海冰、其他）各自换成同一单位、各 3 行 5 列，数据都在，可以做；做好后这里只留一两句加两张图。要先定：这两张新图是否也替换补充材料里的 A3/A4（那样正文和图注都要改）。等你点头再做。
""")

# ---- R3, generalisability: a shorter reply she asked for, as comment lines
rep("""我觉得目前这个回复版本太啰嗦了，你有什么好的建议？
""", """我觉得目前这个回复版本太啰嗦了，你有什么好的建议？
%claude回复：同意你的思路，下面是按它压短的一版，供参考（数字都取自下面现有的回复；冰期海冰扩张要配一两篇重建文献，待核后再加）。原来末尾引的那段 Pellichero 密度范围的话和这条关系不大，建议删掉或挪到给审稿人 1 的回复里。
%Thank you for this question. We agree that open-ocean convection in our pre-industrial state is a model bias, and we now say so in the Discussion. The regime shift, however, does not rest on that bias. Proxy reconstructions agree that glacial winter sea ice was far more extensive than today. Where sea ice covers the ocean it blocks the heat loss to the atmosphere, so the thermal route to dense water weakens, while brine rejection continues. The shift in our simulations follows from this. We also tested that it does not come from open water in the interglacial integrals. Over the sea ice zone alone, the glacial states are 15\\% (LGM) and 17\\% (MIS3) thermally driven at the transformation maximum, against 53--70\\% in the interglacial states. A model that formed its dense water on the shelf would change the density and the origin of the water produced, but not the direction of this shift. We have narrowed our claim accordingly. The paper now presents how the split of surface buoyancy forcing between heat and freshwater responds to changing sea ice cover, not a reconstruction of past bottom water formation rates.
""")

# ---- R3, Gray et al. (2023)
rep("""我们在那一层上是什么响应？
""", """我们在那一层上是什么响应？
%claude回复：是 10 m 风（ECHAM var165，JJA，纬圈平均）。jet 一般指高空急流，建议改成 the latitude of the strongest near-surface westerlies 或 the surface westerly wind belt。Gray et al. (2023) 重建的是海表西风应力峰值的纬度和强度：由南大洋浮游有孔虫 d18O 定出 SST 锋面位置，再借 PMIP 模式里的经验关系推断风的纬度，得到 LGM 相对中全新世偏赤道 4.8°，峰值风应力减弱约 25%，到 0.106 N m-2（LITERATURE_NOTES.md 第 3 节）。我们在同一层（地表纬向风应力 var180，纬圈平均，年平均）：MH 峰值在 47.6°S、0.174 N m-2，LGM 在 49.4°S、0.199 N m-2，即偏极一个网格行（约 1.9°）、增强约 15%。JJA 下 PI 45.7°S、0.164，LGM 49.4°S、0.200。方向和 10 m 风一样，与 Gray 相反，回复的结论不变，只是可以补一句风应力的数（初步，纬圈平均含陆地格点）。
""")

# ---- R3, L25 and L339
rep("""The abstract now refers specifically to the abyssal overturning cell and to abyssal water.
""", """The abstract now refers specifically to the abyssal overturning cell and to abyssal water.

\\claudenote{The reviewer also asked for global overturning and ideal age plots. They are Figures~\\ref{fR:moc} and \\ref{fR:ageglobal} in the reply to the earlier comment on ideal age, and one sentence pointing there would answer that part.}
""")
rep("""and we no longer suggest it in our discussion now.
""", """and we no longer suggest it in our discussion now.

\\claudenote{Language: ``no longer'' and ``now'' say the same thing. Suggested: ``and the Discussion no longer suggests it.''}
""")

open(p, 'w').write(s)
print('ok')
