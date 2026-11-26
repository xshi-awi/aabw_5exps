#!/usr/bin/env python
"""User's 11-26 09:48-10:46 notes, letter side. Usage: python3 edits_1126c_letter.py <response_new.tex>
Numbers: fig_formation_location/numbers.txt, pi/wmt_results (JJA bins), data_moc/moc_5exps_geo.nc
(explore/moc_south_cell_numbers.txt), explore/drift_surface_amoc. Every anchor must match once."""
import sys
p = sys.argv[1]
s = open(p).read()
NOTE = '%claude回复（11-26 10:50）：'


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (n, old[:90])
    s = s.replace(old, new)


# ---------- R13 big heading
rep("%大标题去掉\n", "%大标题去掉\n" + NOTE + "去掉了，这张图（R13）上方那行粗体大标题删了，其余不变。R14 两行图各有一行小标题区分两行，没动。\n")

# ---------- generalisability reply: 4.45 question; her inline note swallowed the line break
rep("Of the transformation into water denser than σ\\textsubscript{2} = 36.93 kg m\\textsuperscript{-3}, 61\\% takes place",
    "Of the winter (JJA) transformation into water denser than σ\\textsubscript{2} = 36.93 kg m\\textsuperscript{-3}, 61\\% takes place")
rep("The sea ice term supplies 4.45 Sv of this 4.48 Sv coastal part.%这个数怎么来的，4，45？4.48？我在wmt profile里面怎么看见绿色的线的数值都很小？\nThe class of the transformation maximum",
    "The sea ice term supplies 4.45 Sv of this 4.48 Sv coastal part.\n%这个数怎么来的，4，45？4.48？我在wmt profile里面怎么看见绿色的线的数值都很小？\n"
    + NOTE + "这几个数是冬季（JJA）、60°S 以南、σ2 = 36.93 这一条等密线上的转换（R12 用 0.02 宽的细 bin），只算陆架陆坡那部分：总 4.48 Sv，其中海冰项 4.45、热项 0.51、其他淡水 -0.48（fig_formation_location/numbers.txt）。整条 36.93 上一共只有 7.4 Sv 穿过（海冰项 4.5、热项 3.5）。正文 Fig. 2 的绿线（海冰项）在 36.8--37.0 那一格约 4.9 Sv，37.0--37.2 约 1.5 Sv，同一格里热项 32 Sv、峰值 70 Sv，所以看起来很小；可到了最密的那一段，总量本身就只剩几个 Sv，其中大半是海冰项。R13 那张是年平均、海冰区、中性密度，冬季析盐和夏季融冰相互抵消，绿线最高才约 1.8 Sv，和这里不是同一个量。为免审稿人也看混，句子里加了 winter (JJA)。另：你这条批注接在句尾，会把换行吃掉，下一句会和 part. 连在一起，我把批注挪到了单独一行。\n"
    "The class of the transformation maximum")

# ---------- "contents ... reads"
rep("%这句语法错误不需要改\n", "%这句语法错误不需要改\n" + NOTE + "这一句我没把握你的意思，没动。现在是 The relevant contents in the Discussion reads:，contents 是复数、reads 是单数，对不上。原来的 The relevant passage in the Discussion reads: 语法是对的（passage reads 没问题）。你要哪个：改回 passage，或者写 The relevant text in the Discussion reads:？\n")

# ---------- push/pull: panel c removed
rep("(c) The two responses scale together across the five states (r = 0.77). (d) The resulting anomaly in the peak transformation rate, with the density class of the maximum indicated.\\label{fR:pushpull}} %图c去掉，那个相关系数没啥用",
    "(c) The resulting anomaly in the peak transformation rate, with the density class of the maximum indicated.\\label{fR:pushpull}} %图c去掉，那个相关系数没啥用\n"
    + NOTE + "去掉了。A9（信里 R15）现在三个 panel：push、pull、WMT 响应；图注和正文 2.4 节那句（含 r = 0.77）都改了，下面引文同步。")
rep("Across the five states the two responses scale together ($r = 0.77$; Supplementary Fig. A9c), and the transformation response follows both (Supplementary Fig. A9d).''",
    "Both responses are positive in all five states, and so is the response of the peak transformation rate (Supplementary Fig. A9c).''")

# ---------- Gray: Results passage removed, Discussion synced
rep("and we state this in both the Results and the Discussion.", "and we state this in the Discussion.")
i = s.index("\n\nThe Results now read:\n\n\\begin{change}\n``We note that the simulated glacial westerlies are displaced poleward")
j = s.index("and the Discussion:\n\n\\begin{change}\n``A related limitation concerns the simulated glacial winds.", i)
s = (s[:i] + "\n\n%这个跟discussion重复，可以删掉呀。\n" + NOTE
     + "删了。修订稿 2.1 节里这段（Gray 对比、风应力、PMIP4）整段删掉，信里这段引文也删了，上面回复里「in both the Results and the Discussion」改成 in the Discussion。\n\nThe Discussion now reads:\n\n\\begin{change}\n``A related limitation concerns the simulated glacial winds."
     + s[j + len("and the Discussion:\n\n\\begin{change}\n``A related limitation concerns the simulated glacial winds."):])
rep("Gray et al. (2023) found that the inferred shift exceeds that of any PMIP3/4 model. In the four PMIP4  LGM simulations, the winter wind maximum",
    "Gray et al. (2023) found that the inferred shift exceeds that of any PMIP3/4 model. In the four PMIP4 LGM simulations (Kageyama et al., 2021), the winter wind maximum")
rep("Moreover, Gray et al., (2023) found no correlation", "Moreover, Gray et al. (2023) found no correlation")
rep("whose mean winds are biased. ''\n\\end{change} %我做了更改\n",
    "whose mean winds are biased.''\n\\end{change} %我做了更改\n" + NOTE
    + "修订稿讨论这一段已按你的版本同步。只动了三处小地方：Gray et al., (2023) 去掉逗号；PMIP4 和 LGM 中间多一个空格；PMIP4 那句补了数据出处 Kageyama et al. (2021)，原来是结果部分引的，那段删了以后讨论里就没出处了。\n")

# ---------- westerlies reply: mechanism moved up from L144, equilibrium in one short paragraph
a = s.index("\\response We thank the reviewer for pressing this point. Our reply above addressed model drift")
b = s.index("\\label{fR:westmech}}\n", a) + len("\\label{fR:westmech}}\n")
moved = s[a:b]
moved = moved.replace("\\response We thank the reviewer for pressing this point. Our reply above addressed model drift, and we have now analysed the winter temperature structure of the atmosphere in all five states (Figure~\\ref{fR:westmech}).",
                      "We have analysed the winter temperature structure of the atmosphere in all five states (Figure~\\ref{fR:westmech}).")
assert moved.startswith("We have analysed")
s = s[:a] + "\\response We now explain the strengthening of the westerlies in our reply to the comment above on why the westerlies intensify in both the glacial and interglacial simulations, together with a new analysis of the winter temperature structure of the atmosphere (Figure~\\ref{fR:westmech}). Section 2.1 of the revised manuscript gives a short explanation.\n" + s[b:]
# repair the L144 comment line, keep her note
rep("\\co挪到前面那个问题下面，这里refer to就行mment{R3}{L144: the westerlies appear to increase under both glacial and last interglacial forcings, why?}\n",
    "\\comment{R3}{L144: the westerlies appear to increase under both glacial and last interglacial forcings, why?}\n%挪到前面那个问题下面，这里refer to就行\n"
    + NOTE + "挪了，机制、2.1 节引文和新图都在上面那条回复里，这里只留一句指过去。你这条批注原来打在 \\comment 这个命令中间，会让编译报错，我把命令补全了，批注放到下一行。\n")
# drop my old claudenote at L144 (its content now sits in the moved reply)
k = s.index("%claude回复（11-26）：认真算了（explore/westerly_mechanism/")
k2 = s.index("\n", k) + 1
s = s[:k] + s[k2:]

EQ = ("On equilibrium, each simulation was integrated for at least 1,000 years, and the PI control for 1,500 years. "
      "Over the 100-year analysis period the trend of global mean SST is +0.02, -0.03, -0.03 and +0.03 K per century in PI, MH, LIG and LGM and +0.06 K per century in MIS3. "
      "The trend of the Atlantic meridional overturning circulation is not significant, except for +1.3 Sv per century in PI (p = 0.04). "
      "These are the two quantities by which the PMIP4 protocol judges equilibrium (Kageyama et al., 2018), and we therefore regard the simulations as close to equilibrium. The Methods now state:\n\n"
      "\\begin{change}\n``Over these 100 years the trend of global mean sea surface temperature is within $\\pm0.03$~K~century$^{-1}$ in PI, the mid-Holocene, the Last Interglacial and LGM and $+0.06$~K~century$^{-1}$ in MIS3. "
      "The trend of the Atlantic meridional overturning circulation is not significant, except for $+1.3$~Sv~century$^{-1}$ in PI ($p = 0.04$). "
      "These are the two quantities by which the PMIP4 protocol judges equilibrium (Kageyama et al., 2018).''\n\\end{change}")
c = s.index("\\response \n%这样吧，审稿人关心的只是why westerlies")
c_note_end = s.index("\n", s.index("%这样吧，审稿人关心的只是why westerlies", c)) + 1
d0 = s.index("The Methods now state:\n\n\\begin{change}\n``Residual drift over the analysis period", c_note_end)
d1 = s.index("\\end{change} %这段不需要这么详细，就在original版本上简单加一个sst/amoc趋势就行\n", d0)
tail_note = "%这段不需要这么详细，就在original版本上简单加一个sst/amoc趋势就行\n"
d1e = d1 + len("\\end{change} ") + len(tail_note)
assert s[c_note_end:d0].strip() == '', repr(s[c_note_end:d0])
s = (s[:c] + "\\response\n" + s[s.index("%这样吧，审稿人关心的只是why westerlies", c):c_note_end]
     + NOTE + "按你说的改了。后面 L144 那一大段（机制、2.1 节引文、新图）整个挪到这里，L144 那里只留一句指回来。平衡只写一段：都至少转了 1000 年（PI 1500 年），给 SST 和 AMOC 趋势，引 Kageyama et al. (2018)，即 PMIP4 第一部分 GMD 11, 1033，DOI 在 Crossref 核过；它 4.1 节原话是 run until the absolute value of the trend in global mean SST is less than 0.05 K per century and the AMOC is stable。MIS3 的 SST 趋势 +0.06 略超 0.05，照实写了。原来的 2 m 气温趋势、JJAS 气温、MH 风与 PI 相同那些细节都删了。\n"
     + moved + "\n" + EQ + "\n" + tail_note
     + NOTE + "Methods 改回了投稿版的写法（PI 1500 年、古气候各 1000 年、分析最后 100 年），只加了上面引的 SST/AMOC 三句和 PMIP4 出处；原来那段 2 m 气温趋势和整段对比（JJAS 气温、MH 风、深渊环趋势）都删了。MIS3 是否「最后 100 年」那条还等你给。\n"
     + s[d1e:])

# ---------- MOC question
rep("%最南边0-4000m，可能是60-70s，那一条是什么，是向下流的水吗？绝对值那里是浅蓝色，anomaly是紫色，跟我们冰期wmt减弱有矛盾吗？\n",
    "%最南边0-4000m，可能是60-70s，那一条是什么，是向下流的水吗？绝对值那里是浅蓝色，anomaly是紫色，跟我们冰期wmt减弱有矛盾吗？\n"
    + NOTE + "那一条是南极边缘一个浅的逆时针小环：表层往南，贴着南极下沉，下面往北回流，所以是向下流的水。它的强度在五个态都差不多，60--70°S 最小值 PI -3.3、MH -3.3、LIG -3.1、LGM -3.5、MIS3 -3.5 Sv，最强处在 160--340 m（data_moc/moc_5exps_geo.nc，explore/moc_south_cell_numbers.txt）。紫色距平主要在 63--67°S 的 1500--2000 m：冰期这个环伸得更深，65°S、1920 m 处 PI +0.8、LGM -1.2 Sv。同时 PI 在 4000 m 左右的深层下沉支（65°S -1.1 Sv）冰期没了（LGM +0.5），55°S 2000 m 以下往北的输出从 -1.5 Sv 变成 0。也就是冰期南极边上下沉的水只到中层，不再进深渊、不往北走，这和深渊环从 11.7 降到 2.3、1.7 Sv 是一回事，和冰期 WMT 减弱不矛盾。另外深度坐标下的纬向平均流函数在南极边上会混进环流（Weddell、Ross 环）的贡献，这个小环不能直接当成表面转换率来读。\n")

open(p, 'w').write(s)
print('response_new.tex: edits applied')
