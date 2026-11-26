#!/usr/bin/env python
"""User's 11-26 07:12-08:22 notes, letter side. Usage: python3 edits_1126b_letter.py <response_new.tex>
Numbers: explore/mld_coastal_oras5/numbers*.txt, fig_formation_location/numbers.txt, data_moc/CLAUDE.md.
Every anchor must match exactly once (or the stated count)."""
import sys
p = sys.argv[1]
s = open(p).read()
NOTE = '%claude回复（11-26 08:30）：'


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (n, old[:90])
    s = s.replace(old, new)


KM2 = ' million km\\textsuperscript{2}'

# ---- R3 main comment: her rewrite note + colour bar note
rep("""reflecting where the model actually forms its dense water. 
%这改：我们的ora5数据的对流基本上都发生在沿岸冰间湖，我们的也有这样的对流，但是比较弱，大部分是发生在weddell那边的open water
%其实如果把colorbar改一下的话，让200m内的那些数也能明显一些，我们沿岸冰间湖对流可以稍微能清楚一下，请你试着看，用什么方法把0-200之间的颜色给弄的深一些。
""", """reflecting where the model actually forms its dense water. In the ORAS5 reanalysis, the deep winter mixed layers near Antarctica lie mostly along the coast, where coastal polynyas form. South of 65°S, 87\\% of the area with a winter mixed layer deeper than 200 m lies within 200 km of the coast. Our simulation also has deep winter mixed layers along the coast, but fewer and shallower ones. Within 200 km of the coast, the area with a winter mixed layer deeper than 200 m is 0.05""" + KM2 + """ in the model and 0.29""" + KM2 + """ in ORAS5. Most of the deep winter mixed layer in our simulation lies offshore, in the open water of the Weddell gyre. South of 65°S, the Weddell gyre holds 87\\% of the model area with a winter mixed layer deeper than 200 m, and the coastal band holds 6\\%. This is open-ocean convection.
%这改：我们的ora5数据的对流基本上都发生在沿岸冰间湖，我们的也有这样的对流，但是比较弱，大部分是发生在weddell那边的open water
%其实如果把colorbar改一下的话，让200m内的那些数也能明显一些，我们沿岸冰间湖对流可以稍微能清楚一下，请你试着看，用什么方法把0-200之间的颜色给弄的深一些。
""" + NOTE + """按你的意思改了。数字是用 ORAS5（1990--2009 JJA）和 PI 各自的原始网格算的（explore/mld_coastal_oras5/）。65°S 以南冬季 MLD 超过 200 m 的面积，ORAS5 有 87\\% 在离岸 200 km 内，模式只有 6\\%，另有 87\\% 在 Weddell 开阔洋；离岸 200 km 内超过 200 m 的面积，模式 0.05、ORAS5 0.29 百万 km²。所以只写了「南极附近」：60--64°S 太平洋扇区（80--120°W）ORAS5 也有大片 200--300 m 的深混合层，不在沿岸。色标改了：0--200 m 每 25 m 一档，占色标三分之二，200 以上是 300、400、600、800，沿岸那一圈现在看得清（ORAS5 最深的点在 Terra Nova Bay 附近，约 590 m）。这张图在稿里是 A12，一起换了，图注加了一句说明色标。另外核数时发现 R2、R11 图注和审稿人 1 那段原来写的「两者最深的都在大西洋扇区、都向南极边缘加深」对 ORAS5 不成立（它最深的在太平洋扇区和沿岸），这两处图注、审稿人 1 的回复和讨论里「same large-scale pattern」那句都改了。
""")

# ---- R1 model limitations: same false pattern sentences
rep("Our pre-industrial simulation shows the same broad pattern as the ORAS5 reanalysis. Both the model and the reanalysis keep shallow mixed layers over most of the open Southern Ocean, deepen towards the Antarctic margin, and place the deepest water of the sector in the Atlantic sector. The obvious difference is in the Weddell gyre interior. In our model simulation,",
    "Over most of the region south of 60°S, the model and the ORAS5 reanalysis both keep the winter mixed layer shallower than 100 m. They differ in where the mixed layer is deep. In ORAS5, the deep winter mixed layers near Antarctica lie mostly along the coast, where coastal polynyas form. Our simulation has fewer and shallower ones there. Within 200 km of the coast, the area with a winter mixed layer deeper than 200 m is 0.05" + KM2 + " in the model and 0.29" + KM2 + " in ORAS5. In our model simulation,")
rep("in the Weddell Sea. We also now state that this configuration has no ice shelf cavities, so basal melt beneath floating ice is absent by construction.\n",
    "in the Weddell Sea. We also now state that this configuration has no ice shelf cavities, so basal melt beneath floating ice is absent by construction.\n" + NOTE + "这段和审稿人 3 主意见那段同步改了。原来「两者都向南极边缘加深、最深的都在大西洋扇区」对 ORAS5 不成立。\n")

# ---- captions of R2 and R11 (same figure, same text)
rep("The large-scale pattern agrees: both are shallow over most of the open Southern Ocean, both deepen towards the Antarctic margin, and both show the deepest water of the sector in the Atlantic sector. The difference is where it matters for this study. The model places",
    "Over most of the region both are shallower than 100 m. They differ in where the mixed layer is deep. The model places", count=2)
rep("and its Weddell sector mean is 128 m against 71 m. Area-weighted",
    "and its Weddell sector mean is 128 m against 71 m. In ORAS5 the deepest winter mixed layers lie along the Antarctic coast, for example near Terra Nova Bay in the western Ross Sea, and near 60--64°S in the Pacific sector between 80 and 120°W. Area-weighted", count=2)
rep("so the bias is concentrated in the convecting patch. We note that the two use different mixed layer definitions",
    "so the bias is concentrated in the convecting patch. The colour scale is finer between 0 and 200 m. We note that the two use different mixed layer definitions", count=2)

# ---- quote of the manuscript ORAS5 sentence (revised L251)
rep("The winter mixed layer of the pre-industrial run has the same large-scale pattern as the ORAS5 reanalysis (Zuo et al., 2019) but is too deep in the interior of the Weddell gyre (Supplementary Fig. A12).",
    "Compared with the ORAS5 reanalysis (Zuo et al., 2019), the winter mixed layer of the pre-industrial run is too deep in the interior of the Weddell gyre and too shallow along the Antarctic coast (Supplementary Fig. A12).")

# ---- generalisability reply: grammar + her 'moreover' note
rep("Surface buoyancy forcing would therefore move towards more brine-driven as the ice cover expands. \n\n%这里可不可以加，moreover, in fig r12, the most dense water （大于36.93） in our model is also in coastal polynya, and is haline dominated. (这里我们用的不是中性密度，因为中性密度只能applied在现代，不能applied到冰期)，而immediate water/mode water是open water/热力主导。\n",
    "Surface buoyancy forcing would therefore become more brine-driven as the ice cover expands. Moreover, the densest water that our pre-industrial state forms at the surface comes mainly from brine rejection near the coast (Figure~\\ref{fR:formloc}). Of the transformation into water denser than σ\\textsubscript{2} = 36.93 kg m\\textsuperscript{-3}, 61\\% takes place on the continental shelf and slope, mostly within 200 km of the coast where the winter ice concentration stays below 0.7, as in coastal polynyas. The sea ice term supplies 4.45 Sv of this 4.48 Sv coastal part. The class of the transformation maximum forms mainly by heat loss over ice-free deep water (69\\% in the interglacial states). We use σ\\textsubscript{2} for this comparison. Neutral density is computed from a reference data set of the modern ocean, so we do not apply it to the glacial states.\n\n%这里可不可以加，moreover, in fig r12, the most dense water （大于36.93） in our model is also in coastal polynya, and is haline dominated. (这里我们用的不是中性密度，因为中性密度只能applied在现代，不能applied到冰期)，而immediate water/mode water是open water/热力主导。\n"
    + NOTE + "加了，数字取自 Fig. R12 的 PI 结果：σ2 ≥ 36.93 的转换 61\\% 在陆架陆坡，其中约四分之三在离岸 200 km 内、冬季冰密集度 < 0.7 的地方（冰间湖一类）；这部分 4.48 Sv 里海冰项占 4.45 Sv，基本全是析盐。中性密度那句也写进去了。mode/intermediate water 那半句没写成「mode water 由开阔洋热力形成」：按 σ2，PI 峰值类（36.6--36.8）和模式自己的底层水（中位数 36.76）同密度，在 R12 这里叫它 mode water 会和 R12 的数字打架；稿件里 intermediate and mode water 的说法用在中性密度下海冰区的热主导带（γ 27.33--27.59），那里成立。所以只写了峰值类主要靠无冰深水的热损失（69\\%）。另把 'move towards more brine-driven' 改成 'become more brine-driven'（语法），讨论那句同改。\n")

# ---- Discussion quote: same grammar fix as the manuscript; her note answered
rep("The surface buoyancy forcing would move towards more brine-driven as the ice concentration increases.''",
    "The surface buoyancy forcing would become more brine-driven as the ice concentration increases.''")
rep("%第一句我去掉了，直接从The shift remains 开始.末尾也去掉了\n",
    "%第一句我去掉了，直接从The shift remains 开始.末尾也去掉了\n" + NOTE + "修订稿讨论这一段已按你改的引文同步，开头两句和末尾关于偏差、距平的三句都删了。距平那层意思还留在上面回复里 On the question whether ... 那一段。\n")

# ---- MOC questions
rep("%下层那个圈减弱，就是age增大的原因吗？\n",
    "%下层那个圈减弱，就是age增大的原因吗？\n"
    + NOTE + "第一问，不矛盾。我们算的 WMT 在 60°S 以南，供的是下层（深渊）环；上层环的指数是 20--70°N、500--4000 m 的最大值，靠北半球（北大西洋）的下沉维持，和 60°S 以南的表面转换是两回事。冰期 60°S 以南转换峰值从 69.9 Sv 降到 47.8 和 56.8 Sv，对应下层环从 11.7 Sv 降到 2.3 和 1.7 Sv，这两者是一致的。上层环冰期变强（18.5 到 20.1、21.5 Sv；1000 m 以下的参照值 16.3 到 19.9、21.5 Sv），大西洋翻转也强 4.6--5.3 Sv。间冰期的上层环指数落在窗口顶边（580 m、21°N），部分量到的是副热带浅层环流（data_moc/CLAUDE.md）。想给审稿人加一句的话可以用：The upper cell is maintained by sinking in the North Atlantic, while the surface transformation south of 60°S feeds the abyssal cell, so the two cells respond separately.\n"
    + NOTE + "第二问，基本是。冰期变老最多的是靠下层环供水的地方（2000 m 以下南大洋 381 到 789、844 年，太平洋 847 到 1291、1314 年），靠上层环通风的大西洋变得少（377 到 439、508 年），这个分布和下层环减弱、上层环维持对得上，回复里 consistent with the abyssal cell weakening while the upper cell is maintained 就是这个意思。两点保留：ideal age 也受混合影响；冰期深海还没平衡（docs/model_equilibrium.md），Methods 把冰期年龄当下界。\n")

open(p, 'w').write(s)
print('response_new.tex: edits applied')
