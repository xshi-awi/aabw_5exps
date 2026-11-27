#!/usr/bin/env python3
"""11-27 pass on revision/response_new.tex (TexPage c9af1a1): the user's 11-27 notes, and the
replies to Reviewers 1 and 2 aligned with the replies to Reviewer 3.
Usage: edits_1127_letter.py <response_new.tex>
All LaTeX goes in raw strings (an earlier pass turned \\t and \\r into control characters).
Every anchor must match exactly once."""
import sys

p = sys.argv[1]
s = open(p, encoding='utf-8').read()
T = '（11-27）：'


def rep(old, new, n=1):
    global s
    c = s.count(old)
    assert c == n, (c, old[:100])
    s = s.replace(old, new)


# ---------------------------------------------------------------- Reviewer 1, model evaluation
rep(r'fresher than observed. We quantify this under model limitations below.', r'fresher than observed.')
rep(r'The reviewer, and Reviewer 3, also asked whether we could compute an observational transformation',
    r'Reviewer 3 also asked whether we could compute an observational transformation')
rep(r'the magnitude of the densification peak agrees well.', r'the magnitude of the densification peak agree well.')

# ---------------------------------------------------------------- Reviewer 1, SAM claims
rep(r'''\end{change} %检查这个是不是跟reviewer 3的回复对齐了？我记得在给reviewer 3的回复中，我们更新过revison里面的类似内容
''', r'''\end{change} %检查这个是不是跟reviewer 3的回复对齐了？我记得在给reviewer 3的回复中，我们更新过revison里面的类似内容
%claude回复''' + T + r'''核过。这段引文和给审稿人 3 L339 那条、审稿人 1 第 29 条的引文一字不差，也和修订稿讨论里的原文一致。你记得的给 R3 更新过的类似内容，应该是讨论里 Gray 那段的末句（SAM 结果只描述模式在有偏差的平均风上叠加风异常时的响应），它也是在收敛 SAM 的说法，我把这句加进了下面。

The Discussion now also states that the westerlies of our glacial simulations shift poleward, whereas proxy reconstructions infer an equatorward shift (see also our reply to Reviewer 3 on the glacial westerlies). The paragraph ends:

\begin{change}
``Thus the SAM results describe how the model responds to a wind anomaly added to a background state whose mean winds are biased.''
\end{change}
''')

# ---------------------------------------------------------------- Reviewer 1, model limitations
rep(r'''Thanks for the comment. We now state the model limitation explicitly. We first diagnosed winter mixed layer depth (Figure~\ref{fR:mld}, now Supplementary Fig. A12), reflecting where the model actually forms its dense water. Over most of the region south of 60°S, the model and the ORAS5 reanalysis both keep the winter mixed layer shallower than 100 m. They differ in where the mixed layer is deep. In ORAS5, the deep winter mixed layers near Antarctica lie mostly along the coast, where coastal polynyas form. Our simulation has fewer and shallower ones there. Within 200 km of the coast, the area with a winter mixed layer deeper than 200 m is 0.05 million km\textsuperscript{2} in the model and 0.29 million km\textsuperscript{2} in ORAS5. In our model simulation, the Weddell sector accounts for 70\% of the area with a winter mixed layer deeper than 400 m south of 55°S, centred near 68°S in the interior of the gyre (see Figure~\ref{fR:mld}a). This is open-ocean convection. In the real ocean, dense water forms through the shelf overflow pathway. We note that this is a limitation shared across the current model generation. Heuzé (2021) found that almost all CMIP6 models form deep and bottom water by open-ocean convection that is too deep, too frequent, or too extensive, with no model showing convincing shelf export in the Weddell Sea. We also now state that this configuration has no ice shelf cavities, so basal melt beneath floating ice is absent by construction.
%这段请和审稿人 3 主意见那段同步
''', r'''Thanks for the comment. We now state the model limitation explicitly. We first diagnosed winter mixed layer depth (Figure~\ref{fR:mld}, now Supplementary Fig. A12), reflecting where the model actually forms its dense water. In the ORAS5 reanalysis, the deep winter mixed layers near Antarctica lie mostly along the coast, where coastal polynyas form. Our simulation also has deep winter mixed layers along the coast, though fewer and shallower. Besides, we also have additional deep convection in our simulation offshore, in the open water of the Weddell gyre. In the real ocean, dense water forms mainly on the shelf and flows down the continental slope. We note that this is a limitation shared across the current model generation. Heuzé (2021) found that almost all CMIP6 models form deep and bottom water by open-ocean convection that is too deep, too frequent, or too extensive, with no model showing convincing shelf export in the Weddell Sea. We also now state that this configuration has no ice shelf cavities, so basal melt beneath floating ice is absent.
%这段请和审稿人 3 主意见那段同步
%claude回复''' + T + r'''已同步。这段按你 11-27 改过的审稿人 3 主意见那段重写（ORAS5 的深混合层多在沿岸，我们沿岸也有但更少更浅，另外 Weddell 开阔洋有对流，Heuzé 2021），删了只在这里出现的 0.05/0.29 百万 km2 和 70% 两个数。没有冰架腔那句留着，因为审稿人 1 问的就是模式局限，按你改的 Heuzé 引文去掉了 by construction。图注换成和 R3 同一张图的图注。图下面新加一小段指向 A13（哪个密度类在哪里形成），结论与给审稿人 3 的一致。
''')
rep(r'''\rfig{0.9}{../letter_figs/figR17_mld_pi_vs_obs.png}{New analysis. Winter mixed layer depth south of 60°S. (a) The pre-industrial simulation. (b) The ORAS5 ocean reanalysis, JJA climatology over 1990-2009. \label{fR:mld}}
''', r'''\rfig{0.9}{../letter_figs/figR17_mld_pi_vs_obs.png}{New analysis. Winter mixed layer depth south of 60°S. (a) The pre-industrial simulation. (b) The ORAS5 ocean reanalysis, JJA climatology over 1990-2009. The color scale is finer between 0 and 200 m. We note that the two use different mixed layer definitions, a 0.01 kg m\textsuperscript{-3} density threshold in ORAS5 and a bulk Richardson number criterion in the model, so the difference is not purely model error.\label{fR:mld}}

To see where the model forms its dense water, we also split the winter transformation south of 60°S by region and density class (new Supplementary Fig. A13, shown as Figure~\ref{fR:formloc} in our reply to Reviewer 3). In the interglacial states the class of the transformation maximum forms mainly by heat loss over ice-free deep water (69\%). The densest classes that each interglacial state produces (3.0--7.4 Sv) form mainly on the continental shelf and slope (61--88\%), most of them within 200 km of the coast. The model therefore does form dense water near the coast, especially in the densest classes, but in the interglacial states this water is small in volume. In the glacial states the whole region south of 60°S is ice covered in winter, and the class of the transformation maximum forms under the ice offshore, mainly by brine rejection.
''')

# sea ice comparison: JJA only, two panels, one paragraph (user note)
rep(r'''We have also compared the simulated sea ice with satellite observations (Figure~\ref{fR:seaice}). The observations are the OSI SAF sea ice concentration climate data record (version 3.0) averaged over 1979–2008, the first 30 years of the satellite era. The model fields are the last 100 years of the pre-industrial run.

\rfig{0.9}{../letter_figs/figR21_seaice_obs.png}{New analysis. Southern Ocean sea ice concentration in the pre-industrial simulation (model years 2300--2399) and in the OSI SAF satellite record for 1979--2008 (sea ice concentration climate data record, version 3.0). (a, b) September, the month with the most ice. (c) Model minus observations in September. (d, e) February, the month with the least ice. In (a, b, d, e) the colours are drawn in the same way as the sea ice in Supplementary Fig. A2. Model and observations are first averaged onto a 1° grid. Cells with less than 1\% ice are left out, and the field is then smoothed with a Gaussian filter 1.5 grid cells wide, which also fills those cells from their neighbours. Values below 10\% are shown in the darkest blue. The dashed orange lines (model) and solid black lines (observations) mark the 15\% ice edge of the unsmoothed fields. (f) Seasonal cycle of the Southern Hemisphere sea ice extent (area with a concentration of at least 15\%, solid lines) and sea ice area (dashed lines), with the range of individual years shaded.\label{fR:seaice}}
%只画jja就行，你直接用supplementary里面海冰的那个文件，那个是需要放在这里的。然后a画海冰，b画差异，其他的不要了。你写的也太多，简单对比一段话就行。请更新。
The model and the observations agree in the timing of the seasonal cycle, with the most ice in September and the least in February. They also agree on the position of the winter ice edge in the Ross Sea and Amundsen–Bellingshausen sector (180°–60°W), where the simulated September edge lies within 0.4° of latitude of the observed one. Elsewhere the model has too little ice. Its September extent is 9.9 million km\textsuperscript{2}, about half of the observed 19.1 million km\textsuperscript{2}, and its February extent is 0.7 against 3.3 million km\textsuperscript{2}, so almost all of the ice melts in summer. Most of the winter deficit is in the Weddell sector. There the simulated September ice edge lies about 10° of latitude south of the observed one, and between 60 and 70°S and 30°W and 30°E the simulated September ice concentration is 8\% against 96\% in the observations. This ice-free part of the Weddell gyre is where the model convects (Figure~\ref{fR:mld}). Too little ice also leaves a larger area of open water for the atmosphere to cool, which is consistent with the strong thermal contribution in our pre-industrial transformation.
''', r'''We have also compared the simulated winter (JJA) sea ice with the OSI SAF satellite record for 1979--2008 (Figure~\ref{fR:seaice}). The two agree on the position of the winter ice edge in the Ross Sea and Amundsen--Bellingshausen sector (180°--60°W), within about 1° of latitude. Elsewhere the model has too little ice. Its winter sea ice extent is 7.9 million km\textsuperscript{2}, about half of the observed 16.5 million km\textsuperscript{2}. Most of the deficit is in the Weddell sector, where the simulated ice edge lies about 9° of latitude south of the observed one and the interior of the gyre is almost free of ice. This is where the model convects (Figure~\ref{fR:mld}).

\rfig{0.9}{../letter_figs/figR21_seaice_obs.png}{New analysis. Winter (JJA) sea ice concentration in the pre-industrial simulation (model years 2300--2399) compared with the OSI SAF satellite record for 1979--2008 (sea ice concentration climate data record, version 3.0). (a) The pre-industrial simulation. (b) Model minus observations. The solid black and dashed orange lines mark the observed and the simulated 15\% ice edge.\label{fR:seaice}}
%只画jja就行，你直接用supplementary里面海冰的那个文件，那个是需要放在这里的。然后a画海冰，b画差异，其他的不要了。你写的也太多，简单对比一段话就行。请更新。
%claude回复''' + T + r'''已更新。图只剩两格、只画 JJA（6--8 月）：(a) PI 海冰，画法和补充材料 A2 的海冰行一样（同一色标、同样平滑），叠了卫星和模式的 15% 冰缘线；(b) PI 减卫星。文字压成上面一段，数字是 6--8 月的：海冰范围 PI 7.9、卫星 16.5 百万 km2；冰缘在 Ross--Amundsen--Bellingshausen 扇区差约 1°，Weddell 扇区偏南约 9°。要你知道的一点：A2 里 PI 那格海冰读的仍是被覆盖的 2090--2099 文件（就是 FS 那件事，先不动），这里用的是论文的 PI 年份 2300--2399，所以 (a) 的 Weddell 环流中间有一块没冰，A2 的 PI 那格是一整圈。要两张看起来一致，得先定 FS（A2 换成正确的 PI 文件）。
''')
rep(r'''Taken together these are systematic biases. That matters for how the results should be read. Because the same bias is present in all five experiments, it largely cancels in the differences between them, which is what the paper reports. Our claim is that the balance between the thermal and haline pathways shifts between climate states, and a bias common to every state cannot generate that shift. The claim leaves aside the absolute transformation rate and the absolute density at which it occurs.
''', r'''These are systematic biases, and they bear on how the results should be read. Our claim is that the balance between the thermal and haline pathways shifts between climate states. The open-ocean convection of the pre-industrial state is a model bias, but the shift does not rest on it. Over the sea ice zone alone, where the open convection area is excluded, the glacial states are 15\% (LGM) and 17\% (MIS3) thermally driven at the transformation maximum, against 53--70\% in the interglacial states (manuscript Fig. 8). Where sea ice covers the ocean it blocks the heat loss to the atmosphere while brine rejection continues, and the shift follows from this. The glacial simulations have biases of their own, so we place more weight on the direction of the glacial minus pre-industrial change than on the absolute transformation rate or the absolute density at which it occurs. Our reply to the third major comment of Reviewer 3 discusses this further.
%claude回复''' + T + r'''这段对齐审稿人 3 第三条。原来写的是同一偏差五个实验都有、在差值里抵消，可冰期没有开阔洋对流，偏差并不相同，这正是审稿人 3 问的「是不是从更偏的 PI 变成了更真实的冰期」。现在和给审稿人 3 的回复用同一套说法：只算海冰区，仍是冰期 15/17% 对间冰期 53--70%，转变跟着海冰走；冰期也有自己的偏差，所以更看重变化的方向。
''')

# ---------------------------------------------------------------- Reviewer 1, specific comments
rep(r'''\response Thank you for catching this. The three abbreviations are now capitalised consistently throughout the manuscript, the figure captions and the supplementary. MIS is written as MIS3 everywhere so that it matches the column headings of the figures.''',
    r'''\response Thank you for catching this. LGM and MIS3 are now capitalised consistently throughout the manuscript, the figure captions and the supplementary, and MIS is written as MIS3 everywhere so that it matches the column headings of the figures. Following Reviewer 3, the text and the captions now spell out the mid-Holocene and the Last Interglacial, and the abbreviations MH and LIG remain only as panel labels in the figures and in Table 1.''')
rep(r'''\end{change}%第一句去掉了。
''', r'''\end{change}%第一句去掉了。
%claude回复''' + T + r'''修订稿摘要同步删了这句。
''')
rep(r'''\response We rewrote it as

\begin{change}
``Further analysis suggests that the Southern Annular Mode modulates transformation by 3--8 Sv between its positive and negative phases in every climate state, working through heat loss when the ocean is open and through brine rejection when it is ice covered.''
\end{change} %？？？
''', r'''\response We rewrote the sentence so that it names the two mechanisms, heat loss when the ocean is open and brine rejection when it is ice covered. The revised sentence is quoted under comment 2 above.
%？？？
%claude回复''' + T + r'''第 2、3 条原来引的是摘要里相邻的两句（第 2 条要过渡句，第 3 条要把 same mechanisms 说清楚）。过渡句删掉以后两条只剩同一句，所以看着重复。现在第 3 条不再重复引文，只说这一句点明了两种机制，引文见第 2 条。
''')
rep(r'``In the last interglacial the Ross sector stands out.', r'``In the Last Interglacial the Ross sector stands out.')
rep(r'''\response We have added this diagnosis to the Results. Figure~\ref{fR:mld} above shows the pre-industrial winter mixed layer depth against ORAS5, and Supplementary Fig. A2 shows it for all five climate states. In the pre-industrial state, the Weddell sector accounts for 70\% of the area with winter mixed layers deeper than 400 m south of 55°S and 97\% of the area deeper than 600 m, centred near 68°S in the gyre interior. The formation region is therefore the open Weddell gyre. This is directly relevant to the model limitation discussed above. This also explains the pattern the reviewer noticed: the Southern Annular Mode heat flux signal is strongest over open water, and in the glacial states the coastal regions where dense water is produced are precisely those insulated by ice, so the surface flux anomalies there are small even though the transformation response is not.''',
    r'''\response We have added this diagnosis to the Results and the Discussion. Figure~\ref{fR:mld} above shows the pre-industrial winter mixed layer depth against ORAS5, and Supplementary Fig. A2 shows it for all five climate states. In the pre-industrial state the deep winter mixed layers lie mainly in the open Weddell gyre, centred near 68°S, and the Weddell sector accounts for 70\% of the area with winter mixed layers deeper than 400 m south of 55°S. We also split the winter transformation by region and density class (new Supplementary Fig. A13, described above under model limitations). In the interglacial states the class of the transformation maximum forms mainly by heat loss over ice-free deep water, most of it in the Weddell sector, while the densest classes form mainly on the continental shelf and slope. In the glacial states the class of the maximum forms under sea ice offshore, and the densest classes form both on the shelf and under the ice further offshore. This also explains the pattern the reviewer noticed. The Southern Annular Mode heat flux signal is strongest over open water, and in the glacial states the regions where dense water is produced lie under the ice, so the surface heat flux anomalies there are small even though the transformation response is not.''')

# ---------------------------------------------------------------- Reviewer 2
rep(r'with densification peaks of +5.0 Sv for us and +4.4 Sv for ORAS5 one density class apart.',
    r'with densification peaks of +5.0 Sv for us and +4.4 Sv for ORAS5.')
rep(r'both are positive almost everywhere and peak one density class apart, but ours is the stronger,',
    r'both are positive almost everywhere, but ours is the stronger,')
rep(r'We also state that resolving these processes would need a considerably finer mesh near the continent, of the order of a few kilometres.',
    r'We also state that resolving these processes would need a considerably finer mesh near the continent, of the order of a few kilometres. Our new analysis of where the model forms each density class (Supplementary Fig. A13, see our reply to Reviewer 3) shows that the densest water of each interglacial state forms mainly on the continental shelf and slope (61--88\%), most of it within 200 km of the coast where polynyas form. The model therefore represents the brine-driven coastal pathway, although in the interglacial states the water it makes there is small in volume.')
rep(r'''\response Thanks for pointing out this. We corrected them. The five experiments and their abbreviations are now introduced at the start of the Results, at first use, in addition to the Methods definition. Reviewer 3 made the same point.''',
    r'''\response Thanks for pointing out this. The five experiments and their abbreviations are now introduced at the start of the Results, in addition to the Methods. Reviewer 3 made a related point, and following that reviewer the text now spells out the mid-Holocene and the Last Interglacial each time. The abbreviations MH and LIG remain only as panel labels in the figures and in Table 1.''')
rep(r'''We have also shortened the section, removing the repetition that Reviewer 1 flagged and cutting the speculative passage about marine proxy reconstruction. What remains in the main text is a compact account of the result, with the supporting composites available to readers who want them.''',
    r'''We have also removed the repetition that Reviewer 1 flagged and the speculative passage about marine proxy reconstruction, and the supporting composites are now in the Supplementary Information for readers who want them.
%claude回复''' + T + r'''对齐时发现一处要你定。审稿人 2 要缩短 SAM 部分，原回复说 We have also shortened the section，可修订稿 2.4 节现在约 1,430 词，投稿版约 730 词，长了将近一倍（主要是给审稿人 1 第 21、22、24 条和审稿人 3 push/pull 加的内容）。我把这句改成只说删了重复和沉积记录那段，不再说缩短。要不要真把 2.4 砍短（比如把风与海冰空间型那段、热通量分区那段各压成一两句），你定。''')

# ---------------------------------------------------------------- Reviewer 3
rep(r'''Because glacial waters are denser overall, we defined the classes separately for each state, as the class of the transformation maximum and as the densest classes that the state produces, beyond which the transformation stays below 10\% of its maximum. %这里没看懂，你是怎么定义的阈值的？
''', r'''Because glacial waters are denser overall, one fixed density threshold cannot be used in all five states, so we defined two classes from the transformation curve of each state. The first is the class of the transformation maximum, the 0.2 kg m\textsuperscript{-3} bin of manuscript Fig. 2 in which the curve peaks. The second is the densest classes that each state produces. To find them we recomputed the transformation in bins ten times finer and took the density on the dense side of the maximum beyond which the transformation stays below 10\% of its maximum, for example σ\textsubscript{2} = 36.93 kg m\textsuperscript{-3} in PI.
%这里没看懂，你是怎么定义的阈值的？
%claude回复''' + T + r'''上面这句改写了，说得更直白。每个态按自己的 WMT 曲线定两个密度类。峰值类就是正文 Fig. 2 曲线最高的那个 0.2 宽的格子（PI 是 σ2 36.6--36.8）。最密类：把曲线用细十倍的格子（0.02）重算，从峰值往密的一侧走，转换率掉到峰值的 10% 以下、之后再也不回升的那条等密线就是阈值，比它更密的都算最密类。PI 是 36.93（这里只有 7.4 Sv，峰值约 78 Sv），MH 36.95，LIG 36.91，LGM 37.89，MIS3 37.47。不用一个固定阈值，是因为冰期的水整体更密，比如 σ2 = 37.0 在 PI 只剩陆架那一点水，在 LGM 却比峰值还轻。
''')
rep(r'most of them within 200 km of the coast where polynya occurs.', r'most of them within 200 km of the coast where polynyas form.')
rep(r'as neutral density can not be applied in glacial states.', r'as neutral density cannot be applied to the glacial states.')
rep(r'we also compared our model with a reanalysis product (ORA5),', r'we also compared our model with a reanalysis product (ORAS5),')
rep(r'against a minimum near -22±5  Sv and a densification peak', r'against a minimum near -22±4 Sv and a densification peak')
rep(r'''its thermal part is confined to a narrow band of intermediate and mode water densities.

%Because''', r'''its thermal part is confined to a narrow band of intermediate and mode water densities.
%claude回复''' + T + r'''-22±5 改成了 -22±4，Pellichero et al. (2018) 原文是 22 ± 4 Sv。

%Because''')
rep(r'''because the simulated Southern Ocean surface is fresher than observed. This is a shared model bias in most climate models.
''', r'''because the simulated Southern Ocean surface is fresher than observed. This is a shared model bias in most climate models.
%claude回复''' + T + r'''这句目前没有出处。能找到的是 Heuzé (2021)：CMIP6 多模式平均的南大洋底层水偏淡 0.02 psu（说的是底层水，不是表层），以及 Toggweiler and Samuels (1995) 说海洋模式里南极附近的盐度偏淡。要不要加引用或改成 many models，你定。审稿人 1 那边同一处我没有加这句，只删了原来那句没兑现的 We quantify this under model limitations below。
''')
rep(r'''Moreover, the densest water that our pre-industrial state forms at the surface comes mainly from brine rejection near the coast (Figure~\ref{fR:formloc}). Of the winter (JJA) transformation into water denser than σ\textsubscript{2} = 36.93 kg m\textsuperscript{-3}, 61\% takes place on the continental shelf and slope, mostly within 200 km of the coast where the winter ice concentration stays below 0.7, as in coastal polynyas. The sea ice term supplies 4.45 Sv of this 4.48 Sv coastal part.''',
    r'''Moreover, the densest water that our pre-industrial state forms at the surface comes mainly from brine rejection near the coast (Figure~\ref{fR:formloc}). In winter (JJA), 7.4 Sv of surface water is transformed into water denser than σ\textsubscript{2} = 36.93 kg m\textsuperscript{-3}. Of this, 61\% (4.5 Sv) takes place on the continental shelf and slope, mostly within 200 km of the coast where the winter ice concentration stays below 0.7, as in coastal polynyas, and almost all of this coastal part comes from the sea ice term. This densest class lies on the steep dense side of the transformation maximum and is narrower than the 0.2 kg m\textsuperscript{-3} density bins of manuscript Fig. 2, so in that figure it is merged with the thermally driven maximum.''')
rep(r'''%可是我怎么看fig2a里面37那里，黑线和红线都是20sv呢（36.93那里更多），绿线不到5.
''', r'''%可是我怎么看fig2a里面37那里，黑线和红线都是20sv呢（36.93那里更多），绿线不到5.
%claude回复''' + T + r'''你看图没看错，差别在格子宽度。正文 Fig. 2 用 0.2 宽的密度格，36.9 那个点是 36.8--37.0 整格的平均，37.1 那个点是 37.0--37.2 整格的平均。用细格看，转换率在这一段掉得像悬崖：36.89 还有 56 Sv，36.91 是 25 Sv，36.93 只剩 7.4 Sv，37.01 只有 2.5 Sv。36.8--37.0 这一格把悬崖上面热主导的几十 Sv 平均了进去，所以图上 36.9 那点总量约 35 Sv、主要是红线；图上把 36.9 和 37.1 两点用直线连起来，37.0 处就读出约 20 Sv，那是连线的效果，真正穿过 37.0 的只有 2.5 Sv。穿过 36.93 的 7.4 Sv 里海冰项 4.5、热项 3.5，陆架那 4.48 Sv 几乎全是海冰项（4.45）。为免审稿人也这样读，上面那段加了一句，说这个最密类比 Fig. 2 的格子还窄，在 Fig. 2 里和热主导的峰值并在同一格。
''')
rep(r'''
\begin{change}
``The size of the two responses differs between the climate states, and it does not separate the interglacial from the glacial states. The Ekman response is 11.2, 8.9 and 12.8~Sv in PI, the mid-Holocene and the Last Interglacial and 8.0 and 10.1~Sv in LGM and MIS3 (Supplementary Fig. A9a). The brine response ranks the states in a similar order (Supplementary Fig. A9b), which gives the correlation noted above. What changes between interglacial and glacial states is the local balance of the anomalous forcing, which we describe next.''
\end{change} %这段可以删掉。而且我们已经不讲correlation了
''', '')
rep(r'Sea ice cannot explain this change, because the LIG ice edge retreats by about 1°',
    r'Sea ice cannot explain this change, because the circumpolar mean LIG ice edge retreats by about 1°')
rep(r'''%lig的机制我没看懂，大白话是啥
%claude回复：
''', r'''%lig的机制我没看懂，大白话是啥
%claude回复''' + T + r'''大白话：LIG 时近日点在南半球冬季附近，6--8 月地球离太阳更近，30--45°S 收到的太阳辐射比 PI 多 26--33 W/m2。热带和副热带的高空（250 hPa）因此暖了约 2 K，南极上空变化不大，于是高空从低纬到极地的温差变大。高空南北温差越大，西风就越强（热成风关系），这个加强一直传到地面，所以西风变强，位置只稍往南挪。海冰不是原因：LIG 环极平均的冰缘只往南退了约 1°，靠南边近地面的温度梯度反而略变弱。MH 的轨道变化方向相同但小得多，高空没有变暖，西风也没变。正文和信里「冰缘退约 1°」都加了 circumpolar mean，免得和 Ross 扇区冰缘退约 5° 那句打架。
''')
rep(r'and the temperature gradient aloft steepens, while the sea ice edge retreats by about 1°.',
    r'and the temperature gradient aloft steepens, while the circumpolar mean sea ice edge retreats by about 1°.')
rep(r'except for +1.3 Sv per century in PI (p = 0.04) which already integrated for 1500 years.',
    r'except for +1.3 Sv per century in PI (p = 0.04), which had already been integrated for 1500 years.')
rep('\nesponse We follow the reviewer', '\n' + r'\response We follow the reviewer')
rep('σ\textsubscript{2} ≈ 36.5--36.8 kg m\textsuperscript{-3}, or γ\textsubscript{n} ≈ 27.3 kg m\textsuperscript{-3}',
    r'σ\textsubscript{2} ≈ 36.5--36.8 kg m\textsuperscript{-3}, or γ\textsubscript{n} ≈ 27.3 kg m\textsuperscript{-3}')
rep(r'''%这里的公式格式有问题，请改
''', r'''%这里的公式格式有问题，请改
%claude回复''' + T + r'''修好了。是我之前的脚本把 \t 当成了制表符，σ、γ 的下标和 kg m-3 的上标都坏了。第 766 行同样原因坏掉的 \response（印成了 esponse）也一并修了。
''')
rep(r'''\comment{R3}{Please be consistent across figures and label MH, LIG, LGM, MIS3 on top of all panels.}''' + ' \n', r'''\comment{R3}{Please be consistent across figures and label MH, LIG, LGM, MIS3 on top of all panels.}

\response Done. The multi-panel figures of the manuscript and the supplement now name the climate state above each column or panel. Reviewer 1 made the same request (specific comment 17), and Figure~\ref{fR:fig3lab} in that reply shows manuscript Fig. 3 as an example.
%claude回复''' + T + r'''这条原来没有回复，按给审稿人 1 第 17 条的回复补了一句。所有多 panel 图是否真的都加了状态名，我这次没有逐张重看，第 17 条的回复已经这样写了。
''')

assert '\t' not in s and '\r' not in s
open(p, 'w', encoding='utf-8').write(s)
print('ok', p)
