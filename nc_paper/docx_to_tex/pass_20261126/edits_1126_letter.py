#!/usr/bin/env python3
"""2026-11-26 pass: letter edits that answer the user's 11-26 notes in response_new.tex
(TexPage bb5f6eb). Quotes of the manuscript are cut programmatically from the edited revised.tex,
with \\ref replaced by the numbers in revised.aux and \\cite by author-year text.

Every anchor must match exactly once; the script refuses to write otherwise.
Usage: python3 edits_1126_letter.py <response_new.tex> <edited revised.tex> <revised.aux>
"""
import json
import re
import sys

path, rev_path, aux_path = sys.argv[1], sys.argv[2], sys.argv[3]
L = open(path, encoding='utf8').read()
R = open(rev_path, encoding='utf8').read()
AUX = dict(re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]*)\}', open(aux_path, encoding='utf8').read()))

CITE = {
    'Zhou2023': 'Zhou et al., 2023', 'Silvano2020': 'Silvano et al., 2020',
    'Skinner2010': 'Skinner et al., 2010', 'Skinner2017': 'Skinner et al., 2017',
    'Pellichero2018': 'Pellichero et al., 2018', 'Gray2023': 'Gray et al., 2023',
    'Lhardy2021': 'Lhardy et al., 2021', 'Heuze2021': 'Heuzé, 2021', 'Heuze2013': 'Heuzé, 2013',
    'Silvano2018': 'Silvano et al., 2018', 'ToggweilerSamuels1995': 'Toggweiler and Samuels, 1995',
    'Koeve2015': 'Koeve et al., 2015', 'Millet2025': 'Millet et al., 2025',
    'werner2016glacial': 'Werner et al., 2016', 'Shi2022': 'Shi et al., 2022',
    'Shi2025GRL': 'Shi et al., 2025', 'Green2022': 'Green et al., 2022',
    'Kageyama2021': 'Kageyama et al., 2021', 'Sime2016': 'Sime et al., 2016',
    'Shi2022JCLI': 'Shi et al., 2022', 'Orsi1999': 'Orsi et al., 1999',
}


def cites(keys):
    parts = []
    for k in keys.split(','):
        k = k.strip()
        assert k in CITE, f'no author-year text for cite key {k}'
        parts.append(CITE[k])
    return '(' + '; '.join(parts) + ')'


def quotify(s):
    s = re.sub(r'\\citet\{([^}]+)\}', lambda m: CITE[m.group(1)].replace(', ', ' (') + ')', s)
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
    """Cut from the sentence starting with `first` to the end of the sentence starting with `last`."""
    p = para(start)
    i = p.find(first)
    assert i >= 0 and p.count(first) == 1, f'first sentence not unique: {first[:60]!r}'
    if last is None:
        j = p.find('. ', i)
        j = len(p) if j < 0 else j + 1
    else:
        k = p.find(last, i)
        assert k >= 0, f'last sentence not found: {last[:60]!r}'
        j = p.find('. ', k)
        j = len(p) if j < 0 else j + 1
    return p[i:j].rstrip()


def change(text):
    return '\\begin{change}\n``' + quotify(text) + "''\n\\end{change}"


def rep(old, new, count=1):
    global L
    n = L.count(old)
    assert n == count, f'anchor found {n} times, expected {count}: {old[:90]!r}'
    L = L.replace(old, new)


NOTE = '%claude回复（11-26）：'

# ================================================================= Reviewer 3, comparison with Pellichero and ORAS5
rep(r"""We did this, and the result is shown in Figure~\ref{fR:wmtR3ab}. Later we will also compare our model against reanalysis product (Figure~\ref{fR:wmtR3cf}).  %这句话improve一下""",
    r"""We did this, and the result is shown in Figure~\ref{fR:wmtR3ab}. Following the reviewer's second suggestion, we also compared our model with a reanalysis product, as described further below (Figure~\ref{fR:wmtR3cf}).
%这句话improve一下
""" + NOTE + """已改写。""")
rep(r"""our transformation shares the structure of theirs (ig.~\ref{fR:wmtR3ab}), with""",
    r"""our transformation shares the structure of theirs (Figure~\ref{fR:wmtR3ab}), with""")
rep(r"""One density class in our pre-industrial state is thermally dominated, i.e., γ = 27.1–27.67 kg m\textsuperscript{-3}. The freshwater term dominates the light classes below γ = 27.1 kg m\textsuperscript{-3} and the dense classes above γ = 27.67 kg m\textsuperscript{-3}.  Thus, the driving component is mixed in our pre-industrial simulation state, and the thermal part of it sits in a narrow band of intermediate and mode water densities instead of across the whole range.""",
    r"""In our pre-industrial state the heat term dominates only between γ = 27.1 and 27.67 kg m\textsuperscript{-3}. The freshwater term dominates the lighter classes below γ = 27.1 kg m\textsuperscript{-3} and the denser classes above γ = 27.67 kg m\textsuperscript{-3}. The forcing in our pre-industrial state is therefore mixed, and its thermal part is confined to a narrow band of intermediate and mode water densities.""")

# ================================================================= Reviewer 3, generalisability
rep(r"""A model that formed its dense water on the shelf would change the density and the origin of the water produced, but not the direction of this shift. %We have narrowed""",
    r"""A model that formed its dense water on the shelf would probably produce denser water in other places, and its surface buoyancy forcing would still move from heat towards brine as the ice cover expands. %We have narrowed""")
rep(r"""The paper now presents how the split of surface buoyancy forcing between heat and freshwater responds to changing sea ice cover, not a reconstruction of past bottom water formation rates.

The relevant passage in the Discussion reads:""",
    r"""The paper now presents how the split of surface buoyancy forcing between heat and freshwater responds to changing sea ice cover, not a reconstruction of past bottom water formation rates.

On the question whether the glacial state may simply be the more realistic one, we would add one point. The glacial simulations have biases of their own, and the sparse proxy data constrain them less well than observations constrain the modern ocean. This is one reason why paleoclimate modelling studies usually interpret the difference between a past state and the pre-industrial state of the same model, since biases shared by both states partly cancel in the difference. We therefore place more weight on the direction of the glacial minus pre-industrial change than on the absolute values of either state, and we consider this direction robust.

The relevant passage in the Discussion reads:""")
rep(r"""%还有一点，审稿人说我们的pi是biased，但是lgm不是，但是我们也不知道lgm到底biased没有。古气候模式研究里，为了消除系统性差异，所以才用跟pi之间的anomaly去做分析，我们还是认为lgm减去pi的这个大方向应该是对的。你觉得如何礼貌地用这句回复一下？
""",
    r"""%还有一点，审稿人说我们的pi是biased，但是lgm不是，但是我们也不知道lgm到底biased没有。古气候模式研究里，为了消除系统性差异，所以才用跟pi之间的anomaly去做分析，我们还是认为lgm减去pi的这个大方向应该是对的。你觉得如何礼貌地用这句回复一下？
""" + NOTE + """回复里加了一段（On the question whether the glacial state ...），讨论那段末尾也加了两句同样的意思，下面的引文已同步。
""")
# the Discussion quote: whole new paragraph from revised.tex
i = L.index('``It is reasonable to ask whether the regime shift is then simply an artefact')
j = L.index('\\end{change}', i)
old_q = L[i:j]
assert old_q.count('%请不要用') == 1
L = L[:i] + '``' + quotify(para('Because the interglacial dense water forms by open-ocean convection in the model')) + "''\n" \
    + L[j:j + len('\\end{change}')] + '\n' \
    + '%请不要用“不是。。。。而是”或者“是，，，，而不是。。。”的句式，这个请加入写作铁律。另外你这段我没看懂。\n' \
    + NOTE + '整段重写了（修订稿讨论同一段同步改）：去掉了 not ... but / rather than 句式，拆成短句；意思是两条理由：只在海冰区里算，冰期与间冰期的差别还在；冰盖本身挡住失热、结冰继续析盐，这在陆架成水的模式里也一样成立。末尾加了你说的距平那两句。写作铁律要改 ~/.claude 里的 my_style，已转给总管加。' \
    + L[j + len('\\end{change}'):]

# ================================================================= Reviewer 3, push and pull
rep(r"""(b) In the same winters it increases the coastal brine input, by 51 and 42 mSv in PI and LIG and by 24 and 31 mSv in LGM and MIS3.""",
    r"""(b) In the same winters it increases the coastal brine input, by 51, 12 and 42 mSv in PI, MH and LIG and by 24 and 31 mSv in LGM and MIS3.""")
rep(r"""Moreover, the Ekman response is largest in the interglacials (11.2 Sv in PI, 12.8 Sv in LIG) and weakest at the Last Glacial Maximum (8.0 Sv), even though the glacial mean wind stress is stronger. Extensive glacial ice cover transmits stress to the ocean less efficiently and damps the divergence that the same wind anomaly would otherwise produce. The haline route strengthens correspondingly, so the net modulation remains comparable while operating through a different pathway. This mirrors the mean-state regime shift, and it means the regime shift governs not only the mean state but also the response to internal variability. %这句我没懂，mis3的Ekman response比mh还大。""",
    r"""The size of the two responses differs between the climate states. It does not separate the interglacial from the glacial states. The Ekman response is 11.2, 8.9 and 12.8 Sv in PI, MH and LIG and 8.0 and 10.1 Sv in LGM and MIS3, and the brine response ranks the states in a similar order. What changes between interglacial and glacial states is the local balance of the anomalous forcing. Under sea ice the freshwater term of the SAM composite is the larger one over 90--94\% of the area, and north of the ice edge the heat term is the larger one over most of the area (Supplementary Figs. A7 and A8). The expanded glacial ice cover therefore shifts the SAM response towards brine release, in the same way as it shifts the mean state.
%这句我没懂，mis3的Ekman response比mh还大。
""" + NOTE + """你说得对，原来那段说冰期 Ekman 响应最弱、因为海冰削弱了风应力传递，数据不支持：Ekman 响应 PI 11.2、MH 8.9、LIG 12.8、LGM 8.0、MIS3 10.1 Sv，并不按间冰期/冰期分组；「海冰削弱传递」也从没算过（Ekman 输运是用 ECHAM 的地表风应力算的，冰区里本来就没有扣除海冰）。信和修订稿第 2.4 节、讨论里这句话都删了，改成只说两条路径的大小不按冰期分组，真正随冰盖变的是局地热项与淡水项谁大（A7、A8 的 90--94% 那组数）。图注 (b) 补上了 MH 的 12 mSv（原来漏了）。""")

rep(r"""The spatial patterns show the mechanism directly (Figure~\ref{fR:windice}). These results are now described in the main text, and the supporting figures are in the Supplementary Information.""",
    r"""Figure~\ref{fR:windice} shows the spatial pattern. The positive phase strengthens the westerly wind stress over the whole circumpolar belt in every state. The sea ice concentration changes mainly near the winter ice edge, and the net winter ice area grows in all five states. Near the coast the concentration hardly changes, because the winter pack there stays close to full cover. The extra brine release near the coast therefore shows up in the freshwater flux (Figure~\ref{fR:pushpull}b) and leaves little trace in this figure.""")
rep(r"""The sea ice response is a dipole, with ice lost near the coast where polynyas open and brine is rejected, and gained further north where the exported ice melts. The dipole sits further north and is substantially stronger in the glacial states.\label{fR:windice}} %这张图到底说明了啥

%在revision中我们做了哪些修改。其他地方也要贴，方便审稿人。
""",
    r"""The westerlies strengthen over the circumpolar belt in every state. The sea ice concentration changes mainly near the winter ice edge, with gains and losses alternating between sectors and a net gain of ice area in every state. The ice edge response lies further north and is about twice as large in the glacial states.\label{fR:windice}}
%这张图到底说明了啥
""" + NOTE + """说实话，原来的描述（近岸减少、往北增加的偶极子）图上看不出来，量过也不成立（explore/sam_windice_check/numbers.txt）：冬季冰密集度 >0.9 的内圈几乎不变（-0.03 到 +0.04 百万 km2），变化都在冰缘带，各扇区有增有减，总冰面积五个态都增加 0.07--0.57 百万 km2；冰期的变化随冰缘在更北边（54--64°S），绝对值之和约为间冰期的两倍。这张图真正能说明的是两点：正 SAM 在五个态都加强了整圈西风应力（push 的来源）；近岸析盐增加（A9b）在密集度上看不出来，因为冬季内圈本来就接近全覆盖，开了口马上又冻上。信、修订稿正文与 A10 图注都按这个改了。如果觉得这张图没有必要，也可以从补充材料里拿掉，告诉我就行。
%在revision中我们做了哪些修改。其他地方也要贴，方便审稿人。
""" + NOTE + """已贴，见下面三段引文（第 2.4 节）；其他几条（L129、L134、L152、L175、L339、L347、冰架）也都补了修订稿原文。

The new text in Section 2.4 reads:

""" + change(para('These composites indicate that the mode acts on dense water formation through two routes at once')) + """

""" + change(para('The spatial patterns are shown in Supplementary Fig.')) + """

""" + change(para('The size of the two responses differs between the climate states')) + """
""")

# ================================================================= Reviewer 3, minor comments
rep(r"""%还是按照审稿人的来吧，正文里面全部用全称，只在图片里面用简称。
\response The abbreviations are now defined at first use in the Results as well as in the Methods. We have kept them after that, because the figures are organised in five columns labelled with these names and spelling the names out each time would make the text much longer. The abstract spells the names out in full.""",
    r"""%还是按照审稿人的来吧，正文里面全部用全称，只在图片里面用简称。
""" + NOTE + """正文、图注和 Methods 里的 MH、LIG 都改成 mid-Holocene、Last Interglacial；只在图里、表 1 里和两处定义（2.1 节开头说明图里标签、Methods 的实验列表）保留简称。PI、LGM、MIS3 没动（审稿人只点了 MH、LIG）。
\response We follow the reviewer's suggestion. The text, the figure captions and the Methods now spell out the mid-Holocene and the Last Interglacial each time. The abbreviations MH and LIG remain only as panel labels in the figures and in Table 1, and the Results and the Methods state what they stand for.""")

rep(r"""\response We agree. We now increase the range of the colorbar to 1000.  We now ...""",
    NOTE + """A2（MLD 与海冰图）重画了：a--c 的 MLD 色标到 1000 m，d、e 保持原来的 0--400 m，图注写明。数据、平滑、海冰那一行和原图完全一样（PI 海冰仍是原来那个文件，FS 先不动）。脚本 fig_mld_seaice/plot_mld_seaice_a2.py，新图 figures/fig02_climate_mld_seaice_jja_mld1000.pdf。
\response We agree. The colour scale of the interglacial mixed layer depth maps (Supplementary Fig. A2a-c, reproduced below) now runs to 1000 m, so that the deepest winter mixed layers in the Weddell and Ross gyres are resolved. The glacial panels keep the 0--400 m scale, because over 99\% of the area south of 55°S their winter mixed layer is shallower than 300 m. The text also reports the area with mixed layers deeper than 600 m and the maximum mixed layer depth, which reaches 991 m in PI. The revised text reads:

""" + change(sentences('In the interglacial states', 'In the interglacial states', 'In PI this convection is concentrated in the Weddell sector')) + """

\\rfig{0.9}{../letter_figs/fig02_mld1000.png}{Revised Supplementary Fig. A2. June-September mean mixed layer depth (a-e) and sea ice concentration (f-j). The mixed layer depth colour scale now runs to 1000 m in the interglacial panels (a-c) and stays at 400 m in the glacial panels (d,e).\\label{fR:mldA2}}""")

rep(r"""\response Rephrased to ``In our model,  dense water formation is predominantly thermally
driven in the interglacial experiments''. %按这个来""",
    r"""\response Rephrased. The sentence now reads:

""" + change(sentences('In our model, dense water formation is predominantly thermally driven', 'In our model, dense water formation')) + """
%按这个来
""" + NOTE + """已按这句改进修订稿第 2.2 节。""")

rep(r"""%其实我们这篇文章本来就是分析的60s以南的所有水，并没有在result部分并没有谈及到aabw，所以审稿人是对的。
%我们仅仅在discussion里面才谈到了aabw
\response The reviewer's intuition is borne out by the density-class analysis described above. Our thermally dominated transformation maximum sits at γ\textsubscript{n} ≈ 27.3 kg m\textsuperscript{-3}, which is intermediate and mode water density rather than bottom water density. We now state in the Results that the interglacial thermal pathway is more relevant to intermediate and mode water formation than to bottom water proper. We are grateful for the suggestion, which turned out to be one of the more useful reframings in the revision.% 重写""",
    r"""%其实我们这篇文章本来就是分析的60s以南的所有水，并没有在result部分并没有谈及到aabw，所以审稿人是对的。
%我们仅仅在discussion里面才谈到了aabw
% 重写
""" + NOTE + """按你的意思重写了：先认同审稿人，再说我们的分析本来就是 60°S 以南所有表层水的转换，结果部分没有单讲 AABW，只在讨论里谈；然后给出热主导极大值的密度，贴两句修订稿原文。
\response We agree with the reviewer. Our analysis covers the transformation of all surface waters south of 60°S, and the Results do not single out bottom water. Bottom water enters only in the Discussion. In the interglacial states the thermally dominated transformation maximum lies at σ\textsubscript{2} ≈ 36.5--36.8 kg m\textsuperscript{-3}, or γ\textsubscript{n} ≈ 27.3 kg m\textsuperscript{-3} inside the seasonal sea ice zone, which are densities of intermediate and mode waters. We now say this in the Results and return to it in the Discussion. The Results now read:

""" + change(sentences('In our model, dense water formation is predominantly thermally driven', 'We note that this thermally dominated maximum')) + """

and the Discussion:

""" + change(sentences('The second factor is the density range', 'This also indicates that the interglacial thermal pathway')))

rep(r"""makes such a reconstruction impractical, and the Discussion no longer suggests it.""",
    r"""makes such a reconstruction impractical, and the Discussion no longer suggests it. The Discussion now reads:

""" + change(sentences('The Southern Annular Mode emerges as a persistent modulator', 'Whether this coupling could be used for proxy reconstruction is unclear.', 'We do not expect it to be recoverable')))
rep(r"""%我们revision里面如何修改的，在这里贴几句话


\comment{R3}{L347""",
    r"""%我们revision里面如何修改的，在这里贴几句话
""" + NOTE + """已贴。原句末尾的 rather than 也按新规矩改掉了。


\comment{R3}{L347""")

rep(r"""\response Thanks for pointing out this. We corrected it. The text now refers to the radiocarbon age of dissolved inorganic carbon.
%我们revision里面如何修改的，在这里贴几句话""",
    r"""\response Thanks for pointing out this. We corrected it. The text now refers to the radiocarbon age of dissolved inorganic carbon:

""" + change(sentences('Finally, we evaluate the simulated deep ocean ventilation ages.', 'Benthic-planktonic')) + """
%我们revision里面如何修改的，在这里贴几句话
""" + NOTE + """已贴。""")


# ================================================================= Reviewer 1 comment 29: same sediment passage as R3 L339
_old_sed = '``Whether this persistent coupling could be exploited for proxy reconstruction is less clear than we previously suggested.'
assert L.count(_old_sed) == 2
while _old_sed in L:
    i = L.index(_old_sed)
    j = L.index('\\end{change}', i)
    L = L[:i] + '``' + quotify(sentences('The Southern Annular Mode emerges as a persistent modulator', 'Whether this coupling could be used for proxy reconstruction is unclear.', 'We do not expect it to be recoverable')) + "''\n" + L[j:]
# ideal age quote: manuscript now says "glacial age increase"
rep(r"""show that the glacial age is concentrated below roughly 2000 m""", r"""show that the glacial age increase is concentrated below roughly 2000 m""")

# ==== PART2 (AABW location, Gray/PMIP, westerlies, drift, ice shelves) ====
# ================================================================= Reviewer 3, Gray et al. 2023 and PMIP4 (explore/pmip_westerlies)
i = L.index('Two considerations bound the consequences. The discrepancy is shared across the model ensemble')
j = L.index('\\comment{R3}{Interestingly the westerlies seem to intensify')
old = L[i:j]
assert old.count('%你可以在/home/a/a270064/cc_projects/historical_zone') == 1
L = L[:i] + r"""We also compared our model with the PMIP4 ensemble. Four PMIP4 models provide Last Glacial Maximum output in the archive available to us (AWI-ESM-1-1-LR, INM-CM4-8, MIROC-ES2L and MPI-ESM1-2-LR). With the same diagnostic, their winter 10 m wind maximum moves by -0.6 to +1.1° between LGM and PI (positive is poleward), and its strength changes by -6 to +0.4\%. None of them comes close to the reconstructed equatorward shift and weakening, which agrees with the finding of Gray et al. (2023) that the inferred shift exceeds that of any PMIP3/4 model. Our model departs further in the opposite direction, with a poleward shift of 3.7° and a strengthening of 8\%. The failure to reproduce the reconstructed change is therefore common to current models, while the size of our poleward shift is particular to our model.

This affects the two parts of our results differently. The mean-state regime shift depends on the area of sea ice and on brine rejection. Gray et al. (2023) found no correlation across the model ensemble between Antarctic sea ice extent and either the latitude of the westerlies or the position of the SST front. A bias in the position of the westerlies therefore need not carry over into the sea ice field on which our main result rests. The Southern Annular Mode results describe how the model responds to a wind anomaly added to a background state whose mean winds are biased. We therefore read them as a description of the mechanism, and we do not use them to reconstruct glacial wind-driven variability.
%请不要用“不是。。。。而是”或者“是，，，，而不是。。。”的句式，这个请加入写作铁律。另外你这段我没看懂。
""" + NOTE + r"""整段重写，去掉了 rather than 和 not ... but，拆成短句。原来两条理由的意思是：平均态的 regime shift 靠海冰面积和析盐，而 Gray 自己发现模式之间海冰范围与西风纬度不相关，所以西风位置的偏差不一定传到海冰上；SAM 那部分是在有偏差的平均风上加扰动，只能当机制看，不能当成冰期风变率的重建。修订稿讨论那段同步改了。
%你可以在/home/a/a270064/cc_projects/historical_zone里面找一下pmip的数据和脚本，计算一下pmip的结果，看看westerlies在lgm相对pi的变化
""" + NOTE + r"""算了（explore/pmip_westerlies/，用 historical_zone 的数据路径，CMIP6 归档里的 PMIP4）。Levante 上有 LGM 输出的 PMIP4 模式只有 4 个：AWI-ESM-1-1-LR、INM-CM4-8、MIROC-ES2L、MPI-ESM1-2-LR。按我们的指标（JJA 纬圈平均 10 m 纬向风极大），LGM 相对 PI 的位移是 -0.6 到 +1.1°（2 个偏极、2 个偏赤道，没有一个超过一个网格行），强度 -6% 到 +0.4%；年平均海洋风应力峰 4 个都略偏赤道（-0.4 到 -1.2°）。都远没有 Gray 的偏赤道 4.8°、减弱 25%。我们的模式偏极 3.7°（抛物线拟合 4.9°）、强 8%，在这 4 个之外，和 AWI-ESM-1-1-LR（-0.6°）也不一样。所以「模式没抓住重建的变化」是共性，「我们偏极这么多」是我们模式自己的。修订稿 2.1 节原来那句 This disagreement is shared across the PMIP ensemble 只对一半，已改；讨论同步。为什么和 AWI-ESM-1-1-LR 不同（模式版本、冰盖边界条件不同）没有分析。PMIP3 的 LGM 风场本地没有，没算。

The Results now read:

""" + change(sentences('The simulated westerlies strengthen in both', 'We note that the simulated glacial westerlies', 'We return to this disagreement')) + """

and the Discussion:

""" + change(para('A related limitation concerns the simulated glacial winds.')) + """

""" + L[j:]

# ==== PART3 (AABW location, westerly mechanism, drift, ice shelves) ====
rep(r"""That paper is about brine released by sea ice on the continental shelf rather than about melting beneath ice shelves.""",
    r"""That paper is about brine released by sea ice on the continental shelf. It does not treat melting beneath ice shelves.""")
rep(r"""Its conclusion that ocean models are too fresh near Antarctica because of errors in their circulation, not because of their surface boundary conditions, applies directly to our model.""",
    r"""It traces the fresh bias of ocean models near Antarctica to errors in their simulated circulation, which applies directly to our model.""")
# ================================================================= Reviewer 3 main comment: where each density class forms (fig_formation_location)
rep(r"""%最后我们再强调，我们的分析区域是整个南极，60s以南，不仅仅包含aabw，还包含其他密度区间的各类水团转换。但是aabw也是一个非常有趣并且值得分析的课题，不过，由于冰期的密度整体高，无法定义aabw前体水的密度阈值。所以aabw这一块的regime shift难以单独分离出来。。
""",
    r"""%最后我们再强调，我们的分析区域是整个南极，60s以南，不仅仅包含aabw，还包含其他密度区间的各类水团转换。但是aabw也是一个非常有趣并且值得分析的课题，不过，由于冰期的密度整体高，无法定义aabw前体水的密度阈值。所以aabw这一块的regime shift难以单独分离出来。。
%claude回复（11-26）：明白，按你的思路做了（fig_formation_location/）。把论文 60°S 以南的冬季表面 WMT 按区域（陆架陆坡 = 海底浅于 1000 m、深水对流区、冰下深水、无冰深水，另按离岸 200 km）和密度类拆开，五个态的总曲线与论文逐 bin 一致。你提的 26.5/27 我用 σ0 试了（本机没有中性密度库）：PI 冬季 60°S 以南的水几乎都比 26.5 密，27.0 正好在峰值附近、61% 在离岸无冰区，冰期两条线都比所有表层水轻，所以固定阈值分不开；改成每个态按自己的曲线定两类：峰值所在的类，和最密的尾部（转换率降到峰值 10% 以下的那一段）。结果：间冰期峰值类 69% 在无冰深水（热损失），陆架 20--24%；最密尾部 61--88% 在陆架，多在离岸 200 km 内、冬季冰没盖满的地方，也就是冰间湖一类的地方；和 PI 底层水一样密的水只有 13% 来自陆架。冰期峰值类 89--91% 在冰下深水（析盐），最密尾部 53--59% 在陆架；能达到冰期底层水密度的表层水只有 0.7--1.0 Sv，全在陆架。所以模式也有冰间湖造水，但间冰期量小，模式的底层水是开阔洋造的那种密度，这正是审稿人说的偏差；冰期则更像真实海洋的陆架路径。mode water 这个词没用错，稿件里说的是 intermediate and mode water 的密度范围。新图放进了补充材料（A13），讨论加了一段，下面回复里引了。
""")
rep(r"""\response Thanks for the comment. We do not believe shelf processes are overlooked in the real ocean, and we have restructured the discussion so that the model's convection behaviour is stated at the outset rather than left implicit.""",
    r"""\response Thanks for the comment. We do not believe shelf processes are overlooked in the real ocean, and we have restructured the Discussion so that the model's convection behaviour is stated at the outset.""")
rep(r"""This is open-ocean convection, not the shelf overflow pathway that operates in the real ocean as the reviewer supposed.""",
    r"""This is open-ocean convection. In the real ocean, as the reviewer notes, dense water forms on the shelf and flows down the continental slope.""")
rep(r"""so the bias is concentrated in the convecting patch rather than spread evenly. We note that the two use different mixed layer definitions, a 0.01 kg m\textsuperscript{-3} density threshold in ORAS5 and a bulk Richardson number criterion in the model, so the point-by-point difference is not purely model error.\label{fR:mldR3}}""",
    r"""so the bias is concentrated in the convecting patch. We note that the two use different mixed layer definitions, a 0.01 kg m\textsuperscript{-3} density threshold in ORAS5 and a bulk Richardson number criterion in the model, so the point-by-point difference is not purely model error.\label{fR:mldR3}}""")
rep(r"""Our new comparison with satellite sea ice, shown in the reply to Reviewer 1""",
    r"""To see how much dense water the model makes in the open ocean and how much near the coast, we split the winter transformation south of 60°S by region and density class (Figure~\ref{fR:formloc}, new Supplementary Fig. A13). The totals reproduce the published transformation curves exactly. Because glacial waters are denser overall, we defined the classes separately for each state, as the class of the transformation maximum and as the densest classes that the state produces, beyond which the transformation stays below 10\% of its maximum.

In the interglacial states, the class of the transformation maximum (σ\textsubscript{2} = 36.4--36.8 kg m\textsuperscript{-3}, 69--81 Sv) forms mainly by heat loss over ice-free deep water (69\%), and the continental shelf and slope, where the bottom is shallower than 1000 m, contribute 20--24\%. The densest classes (3.0--7.4 Sv) form mainly on the shelf and slope (61--88\%), most of them within 200 km of the coast where the winter pack ice is open. Water as dense as the simulated PI bottom water is made mostly offshore, with 13\% from the shelf. The model therefore does form dense water near the coast. In the interglacial states this water is small in volume, and the simulated bottom water takes the density of the water made offshore. This is the model bias the reviewer describes. In the glacial states the whole region south of 60°S is ice covered in winter. The class of the transformation maximum forms under the ice offshore (89--91\%), mainly by brine rejection, and the densest classes form on the shelf (53--59\%) and under the ice further offshore. Only 0.7--1.0 Sv of surface water reach the density of the glacial bottom water, and all of it forms on the shelf.

\rfig{0.95}{../letter_figs/figS_formation_location.png}{New Supplementary Fig. A13. Where the simulated winter (JJA) surface transformation south of 60°S takes place. (a) Surface density flux into the densest classes of each state (threshold below each panel), on a logarithmic scale. (b) Surface density flux across the class of the transformation maximum. (c) Transformation in the class of the maximum, split into the continental shelf and slope (bottom shallower than 1000 m), deep water with a winter mixed layer deeper than 400 m, deep water under sea ice and ice-free deep water. (d) The same split for the densest classes. (e) The densest classes split by distance from the coast, with the part within 200 km of the coast divided at a winter sea ice concentration of 0.7.\label{fR:formloc}}

We also stress that our analysis covers the transformation of all surface water south of 60°S. This includes mode and intermediate waters as well as the precursors of bottom water. Bottom water is an important part of this, and we discuss it in the Discussion. Its own regime shift is hard to isolate, however, because glacial waters are denser overall and no single density threshold separates the bottom water source in all five states. For example, σ\textsubscript{2} = 37.0 kg m\textsuperscript{-3} selects only shelf water in PI (2.5 Sv), but in LGM it lies below the density of the transformation maximum. The new Discussion paragraph reads:

""" + change(para('To see where each density class is produced')) + r"""

Our new comparison with satellite sea ice, shown in the reply to Reviewer 1""")

# ================================================================= Reviewer 3, westerlies intensify / drift (explore/westerly_mechanism, explore/drift_surface_amoc)
i = L.index(r"""\response The two intensifications have different origins. The LIG strengthening follows that period's orbital configuration""")
j = L.index('The Methods now state:', i)
L = L[:i] + r"""%claude回复（11-26）：Methods（第 4.2 节）现在写的是：PI 积分 1500 年，大气从 AMIP、海洋从 WOA 起步；mid-Holocene、Last Interglacial 从平衡的 PI 分叉；LGM 和 MIS3 were initialized from a previous glacial state (Werner et al., 2016)；古气候实验都积分 1000 年。没写大气冷启动，也没写那个旧冰期态转了 1500 年；要补的话请给出处。drift 按你说的只谈表层和 AMOC（explore/drift_surface_amoc/）：全球 SST 趋势 -0.03 到 +0.06 K/百年（只有 MIS3 略超 0.05）；大西洋翻转环流趋势 -0.6 到 +1.3 Sv/百年，只有 PI 的 +1.3 勉强显著（p=0.04），和年际标准差 1.0--1.5 Sv 相当，冰期比 PI 强 4.6--5.3 Sv；深层环流趋势 -1.0 到 +0.6 Sv/百年，冰期减弱约 9 Sv。回复里只写这两样，Methods 里原来 2 m 气温那几句保留（LIG、MIS3 超标照写），后面加了 SST 和 AMOC 两句。原来那句 two orders of magnitude 不对（0.1 K 对 4--5 K 约 40 倍），改成 more than an order of magnitude。信里原来的 with the final 100 years analysed in each case 删了：MIS3 的输出到 2616 年、分析期是 2500--2599，不是最后 100 年（log/OPEN.md 里等你定的那条），Methods 那句 We analyze the final 100 years of each run 也还等你定。
\response The two intensifications have different origins, which we analyse in our reply to the comment on L144 below. In short, the glacial strengthening follows the expanded sea ice, which steepens the low-level temperature gradient on the poleward side of the westerlies. The LIG strengthening follows the orbital forcing, which warms the tropical upper troposphere.

On drift, we examined the ocean surface and the Atlantic overturning over the 100-year analysis period of each simulation. The trend of global mean SST is +0.02, -0.03, -0.03, +0.03 and +0.06 K per century in PI, MH, LIG, LGM and MIS3, against glacial minus PI differences of -1.8 and -1.5 K. The trend of the Atlantic overturning, taken as the maximum of the Atlantic streamfunction between 500 and 4000 m depth and between 20 and 70°N, is +1.3, -0.4, -0.6, +0.9 and +0.8 Sv per century. Only the PI value is marginally significant (p = 0.04). These trends are comparable to the year-to-year variability of 1.0--1.5 Sv and small against the glacial strengthening of 4.6--5.3 Sv relative to PI. The trends of the abyssal cell lie between -1.0 and +0.6 Sv per century, against a glacial weakening of about 9 Sv. Drift therefore cannot produce the differences between the climate states that we discuss. In addition, MH is indistinguishable from PI in both the latitude (45.7°S in both) and the strength (5.83 against 5.84 m s\textsuperscript{-1}) of the wind maximum. Since MH branches from the same equilibrated PI state and was integrated for the same length as the other paleo experiments, a drift explanation would have to produce a signal in LIG, LGM and MIS3 while leaving MH unchanged, which we think is implausible. The PI control was integrated for 1500 years and all paleo simulations for 1000 years.

""" + L[j:]
i = L.index('The Methods now state:', i)
k = L.index('\\begin{change}', i)
j = L.index('\\end{change}', k)
_m1 = sentences('The PI control run was integrated for 1,500 years', 'Residual drift over the analysis period', 'The Last Interglacial and MIS3 values exceed')
_m2 = para('The glacial-interglacial differences examined here are large by comparison')
L = L[:k] + '\\begin{change}\n``' + quotify(_m1 + ' ' + _m2) + "''\n" + L[j:]

rep(r"""%前面哪里回答了，前面只回答了平衡问题。这里你需要研究sst gradient （不过海冰扩张，sst gradient应该往赤道移动，我很不理解，，，请你认真对待这个问题，好好研究一下。）
\response We answered this the above discussion.""",
    r"""%前面哪里回答了，前面只回答了平衡问题。这里你需要研究sst gradient （不过海冰扩张，sst gradient应该往赤道移动，我很不理解，，，请你认真对待这个问题，好好研究一下。）
%claude回复（11-26）：认真算了（explore/westerly_mechanism/，从 ECHAM 模式层插到 13 个气压层，100 年 JJA 平均）。你的直觉对一半：冰期海冰扩张确实把最强的近地面温度梯度往赤道推了，从约 65°S 到 60--62°S，还变陡 24--29%；但它仍在西风极大以南 10--12°。所以温度梯度加强的地方在西风带的极地一侧，那里的斜压性（850--500 hPa 的 Eady 增长率）在 45--65°S 增加 21--28%，风暴路径和由涡动驱动的西风就往南、变强。高空热带比地面冷得多（LGM 250 hPa -5.4 K，850 hPa -3.0 K），中纬度高空温度梯度变弱，也有利于西风往极地走。开阔洋的 SST 锋在五个态都在 38.5--40.5°S，在西风以北，不是控制因素。LIG 不是海冰的原因（冰缘反而后退约 1°、南边斜压性略减），而是轨道强迫：JJA 日射在 30--45°S 增加 26--33 W m-2，热带高空增暖 2.3 K，高空温度梯度变陡，所以 LIG 主要是变强、位移很小，而且集中在太平洋扇区（原因没再追）。五个平衡态分不开海冰、冰盖高度和温室气体各自的作用，要大气单独敏感性试验。回复、新图 R（figR23_westerly_mechanism.png）和修订稿 2.1 节都按此写了，原来那句 following the steepened meridional temperature gradient imposed by the expanded ice sheets and sea ice 换成了具体的机制和数。
\response We thank the reviewer for pressing this point. Our reply above addressed model drift, and we have now analysed the winter temperature structure of the atmosphere in all five states (Figure~\ref{fR:westmech}).

In the glacial states the change follows the expanded sea ice. The winter ice edge moves from 66°S in PI to 55--57°S. One might expect the westerlies to follow the temperature front towards the equator. The strongest near-surface temperature gradient does move north with the ice edge, from about 65°S to 60--62°S, and it becomes 24--29\% steeper (Figure~\ref{fR:westmech}c). It still lies 10--12° south of the wind maximum, however, and the SST front of the open ocean stays between 38.5 and 40.5°S in every state, north of the westerlies. The steeper gradient therefore raises the baroclinicity on the poleward side of the westerlies. The Eady growth rate between 850 and 500 hPa, a standard measure of the potential for storm growth, increases by 21--28\% between 45 and 65°S (Figure~\ref{fR:westmech}e). The eddy-driven westerlies move towards this region and strengthen. Aloft, the tropical upper troposphere cools more than the surface, by 5.4 K at 250 hPa against 3.0 K at 850 hPa in LGM, which weakens the upper-level temperature gradient in mid-latitudes (Figure~\ref{fR:westmech}f). This also favours stronger westerlies further south. The shift is larger aloft than at the surface. At 850 hPa the wind maximum moves by 6.4° in LGM, because the stable layer over the sea ice weakens the near-surface wind relative to the wind above it.

In LIG the change comes from the orbital forcing. Winter (JJA) insolation increases by 26--33 W m\textsuperscript{-2} between 30 and 45°S. The tropical upper troposphere warms by 2.3 K at 250 hPa, and the upper-level temperature gradient in mid-latitudes steepens (Figure~\ref{fR:westmech}f). Sea ice cannot explain this change, because the LIG ice edge retreats by about 1° and the low-level baroclinicity south of 55°S weakens slightly. The westerlies therefore strengthen with only a small poleward shift. The strengthening is largest in the Pacific sector, which points to a zonally asymmetric response that we have not traced further. MH receives a smaller version of the same insolation change, with no upper-tropospheric warming and no change in the westerlies.

Five equilibrium states cannot separate the roles of sea ice, ice sheet height and greenhouse gases. This would require atmosphere-only sensitivity experiments, which are beyond the scope of this study. We have added a short explanation to Section 2.1:

""" + change(sentences('The simulated westerlies strengthen in both', 'In the glacial states the change follows the expanded sea ice.', 'Five equilibrium states cannot separate')) + r"""

\rfig{0.9}{../letter_figs/figR23_westerly_mechanism.png}{New analysis. Austral winter (JJA) zonal means of the 100-year climatologies of the five states. (a) 10 m zonal wind, with dots at the maximum. (b) 250 hPa zonal wind. (c) Meridional gradient of the surface temperature over ocean grid boxes, including the sea ice surface (thick lines), and of the sea surface temperature of the ocean model (thin lines). (d) Meridional temperature gradient at 850 hPa. (e) Eady growth rate between 850 and 500 hPa. (f) Meridional temperature gradient at 250 hPa. The coloured ticks at the top mark the 15\% winter sea ice edge of each state.\label{fR:westmech}}""")

# ================================================================= Reviewer 3, ice shelves (explore/drift_surface_amoc/antarctic_runoff_check.txt)
rep(r"""\response Unfortunately, just as most other models, the model configuration does not include ice shelf cavities, so there is no ocean beneath the floating ice shelves to show. The white areas are genuinely outside the model domain. We have stated this in the Methods and figure captions, and we list it among the limitations, since the absence of cavities also means basal melt is not represented. Like in most climate models, this freshwater source is provided as river runoff.""",
    r"""%claude回复（11-26）：核过（explore/drift_surface_amoc/antarctic_runoff_check.txt）：冰盖上的降水减蒸发由 JSBACH 冰川格点全部转成径流，HD 模型送到海岸，以液态径流进 FESOM（PI 60°S 以南约 0.085 Sv，冰期约 0.07 Sv）；FESOM 的 use_landice_water、use_iceberg_water 都关着，也没有冰架腔。所以你那句意思对，只是这股径流替代的不只是底部融化，也包括冰山崩解，已改写；「大多数模式」没查文献，写成 many。原回复说图注里也写了没有冰架腔，其实图注里没有，改成 Methods 和讨论。修订稿 Methods 和 2.3 节（审稿人 L195 那段）也补了这一点，L195 下面的引文跟着更新。
\response Unfortunately, like most other climate models, our model configuration does not include ice shelf cavities, so there is no ocean beneath the floating ice shelves to show. The white areas are outside the model domain. We have stated this in the Methods and the Discussion, and we list it among the limitations, since the absence of cavities also means that basal melt is not represented. As in many climate models without ice shelf cavities, the freshwater that leaves the ice sheet returns to the ocean as liquid runoff at the coast. The Methods now read:

""" + change(sentences('We perform five equilibrium simulations', 'No prescribed ice sheet meltwater flux is applied', 'This runoff stands in for')))

# L195 quote: regenerate from the manuscript (2.3 now mentions the runoff)
i = L.index('``In the present-day ocean, meltwater derived from the Antarctic ice sheet')
j = L.index('\\end{change}', i)
L = L[:i] + '``' + quotify(sentences('The sea ice contribution shows a dipole in space.', 'In the present-day ocean, meltwater derived', 'The freshwater that leaves the ice sheet returns')) + "''\n" + L[j:]


# ---------------------------------------------------------------- not-but sweep (user rule 11-26), reviewed list in notbut_1126.json
import os
_here = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(_here, 'notbut_1126.json'), encoding='utf8'))
skipped = []
for p in P['letter']:
    n = L.count(p['old'])
    if n == 1:
        L = L.replace(p['old'], p['new'])
    else:
        skipped.append((n, p['old'][:70]))
# the same manuscript rewrites inside older quotes of the manuscript
CITE.update({'Orsi1999': 'Orsi et al., 1999', 'Weber2007': 'Weber et al., 2007', 'Wainer2012': 'Wainer et al., 2012',
             'Shi2022JCLI': 'Shi et al., 2022', 'Abernathey2016': 'Abernathey et al., 2016'})
n_q = 0
for p in P['revised']:
    try:
        qo, qn = quotify(p['old']), quotify(p['new'])
    except AssertionError:
        continue
    if qo in L:
        L = L.replace(qo, qn)
        n_q += 1
print(f'not-but sweep (letter): {len(P["letter"]) - len(skipped)} applied, {len(skipped)} skipped; {n_q} manuscript rewrites carried into quotes')
for s_ in skipped:
    print('   skipped', s_)

# ---------------------------------------------------------------- MH and LIG spelled out inside the manuscript quotes (same rules as revised.tex)
sys.path.insert(0, _here)
import mhlig
L = re.sub(r'(\\begin\{change\}\n)(.*?)(\n\\end\{change\})', lambda m: m.group(1) + mhlig.spell_out(m.group(2)) + m.group(3), L, flags=re.S)

open(path, 'w', encoding='utf8').write(L)
print('letter edits OK')
