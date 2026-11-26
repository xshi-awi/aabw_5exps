# 模式是否达到平衡：spin-up、全球地表温度漂移、深海趋势

问题来自审稿人 3：西风增强会不会是没平衡的产物。结果对我们不全是好消息。11-26 用户定 Methods 只写 SST 与 AMOC（见文末），下面的 2 m 气温数字只留作档案。

## spin-up 长度

由 GRL 的 Text S1 确认（用户上传的 15 页完整 supplementary，应为 `nc_paper/SUBMISSION/grl_supplementary.pdf`）：PI 1,500 年，4 个古气候实验各 1,000 年，取最后 100 年，与投稿版一致。Text S1 还独立验证了 Table 1 的五个 CO₂ 值和 GLAC-1D 21k/38k 的网格分配。先前以为几篇文献互相矛盾是误判：JCLI 的 1000 年只讲它自己那组实验，CP 的 1300 年只讲它自己的 PI 和 LGM。

## 全球地表温度漂移

判据取自 Shi et al. 2022 JCLI 原文：最后 100 年全球平均地表温度趋势不超过 ±0.05 K/century。用 var167（2 m 气温，含陆地）实测：PI +0.019，MH −0.033，**LIG −0.063**，LGM +0.040，**MIS3 +0.101**。LIG 和 MIS3 超标。

11-25 版 Methods 曾五个数全列、承认超标并给限定论证（冰期与间冰期信号约 4 K；MH 西风与 PI 一致），11-26 删掉。数据 `drift_check/{exp}_t2m.nc`，详表见 `nc_paper/NEW_ANALYSIS_RESULTS.md` §8。2 m 气温含陆地，SST 只覆盖海洋，两者不能互换；现在 Methods 用的是 PMIP4 的 SST 判据。

## 深海（2000 m 以下）趋势

`calc_deep_ocean_trend.py` 读 `production/{exp}_age/outdata/fesom/` 逐年 3D 温盐（每个实验 396–616 年，共 2300 多个文件），体积加权，输出 `deep_ocean_trend.npz`；出图 `plot_deep_trend.py` → `figures/figR10_deep_trend.pdf`。

| exp | 分析期趋势 K/century | 后半段/前半段趋势比 | 整段记录深海升温 |
|-----|------|------|------|
| PI | 0.009 | 0.41 | - |
| MH | 0.016 | 0.18 | - |
| LIG | 0.007 | 0.13 | - |
| LGM | 0.113 | 1.08 | +0.58 K |
| MIS3 | 0.160 | 1.37 | +0.89 K |

间冰期深海已经平了而且还在减速；两个冰期没有，MIS3 甚至在加速。原因是冰期实验从更早的冰期态初始化，而不是从 PI 分叉。Methods 写明影响范围：ideal age 本来就声明是下界；表面 WMT 诊断是对表面强迫的年际响应，不受慢速深海调整影响。

## 2026-11-26 补充：SST 与 AMOC 漂移（审稿人 3，用户要求只谈表层和 AMOC）

`explore/drift_surface_amoc/numbers.txt`，100 年分析期，趋势按自相关修正标准误。全球 SST −0.03 到 +0.06 K/百年（只有 MIS3 的 +0.060 略超 0.05）；大西洋翻转环流（大西洋流函数 500–4000 m、20–70°N 最大值）+1.3、−0.4、−0.6、+0.9、+0.8 Sv/百年，只有 PI 勉强显著（p=0.04），年际标准差 1.0–1.5 Sv，冰期比 PI 强 4.6–5.3 Sv；深层环流 −1.0 到 +0.6 Sv/百年，冰期减弱约 9 Sv。MH、LIG 与 PI 的 AMOC 差（+0.6、+1.6 Sv）和漂移同量级，不要解读。Methods 里原有的「two orders of magnitude」不对（0.1 K 对约 4 K 是 40 倍），11-26 改成 more than an order of magnitude。

## 2026-11-26 11:00 用户定：Methods 只留 SST/AMOC

用户觉得漂移写得太细、「越描越黑」。Methods 改回投稿版（PI 1500 年，古气候各 1000 年，分析最后 100 年），只加三句：SST 趋势四个态在 ±0.03 K/百年内、MIS3 +0.06；AMOC 趋势除 PI（+1.3 Sv/百年，p = 0.04）外不显著；这是 PMIP4 判断平衡用的两个量，引 Kageyama et al. 2018（GMD 11, 1033，ref.bib 键名 `Kageyama2017`，DOI Crossref 核过）。原文 4.1 节：run until the absolute value of the trend in global mean SST is less than 0.05 K per century and the AMOC is stable。2 m 气温趋势、JJAS 气温、MH 风与 PI 相同、深渊环趋势那一段都从 Methods 和信里删了，数字仍在本文件上面几节。

