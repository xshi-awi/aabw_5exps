# 哪份是投稿版：证据（2026-11-27 14:20 她说「nc-main 是投稿的正式版本，比你有的那个还要再新的」后核查）

两个候选：
- A = TexPage `original_submission/nc-main_submitted.tex`；
- B = 本地 `nc_paper/nc-main.tex`，文件时间 2026-05-18 08:04，09-08 随修订进 git（758dfa6），也就是 `build/submitted_original.tex`。

两份有 35 处不同。

## 证据

1. **审稿人引的行号。** B 原来去掉了行号，按 A 的导言补回 `lineno` 后两份都编译，逐条对 30 处审稿人行号引用，以审稿意见里点名的词句为准，结果在 `consistency/provenance_line_test.txt`。
   - 落在所引行 ±1 行内：A 是 27/30，B 是 10/30。
   - 从引言第 3 段起 B 整体错开 3–5 行，例如 moer 在 125 行、Here 在 129 行、only secondarily 在 167 行。
   - 剩下 3 处两份都没对上，是挑的词不够准。
2. **审稿人批评的原句只在 A 里。** R3 第 L347 条说「not the water, the carbon in the water」。
   - A 的 347 行是「deep water circulating around Antarctica was more than two times older than today」。
   - B 里没有这句，写的是「glacial deep ocean radiocarbon ages increased globally」。
3. **09-08 拿到的投稿 PDF 就是 A。** `nc_paper/submitted.pdf`（31 页，09-08 17:56 生成，REVISION_PLAN.md 把它列为投稿 PDF）和 A 编译出的文字逐词比，只差作者块：submitted.pdf 是匿名版，没有作者和单位。
   - 和 B 比，有 109 处成块的差别，相似度 0.90；和 A 比是 0.996。
   - 35 处不同里，submitted.pdf 有 47 个片段跟 A，3 个跟 B。
4. **A 比 B 新。** 本地 `nc_paper/Ref.bib`（2026-05-18 06:18）逐字节是 A 的 `Ref.bib` 的开头，A 的在末尾多追加了 5 条：sime2013southern、chadwick2022compilation、volker2013responses、mandal2021roles、knorr2021salty。正好是 A 正文新加的两句引的文献，B 正文里没有。这是在 B 的基础上加文献的写法，反过来（B 由 A 删掉末尾 5 条得到）很不自然。
5. **作者块。** A 是 4 位作者，Shi、Liu 两位通讯作者带邮箱；B 是 5 位，多 Hu Yang。修订稿 09-08 从 B 起步时也是 5 位，09-23 她在网页上把作者块改成和 A 一样（git 8967a74「adopt his web edits」），同一天 A 上传到 TexPage。

## 结论

审稿人审的是 A。A 也比 B 新（B 是 05-18 的稿，A 在它上面补了引言两句和 5 篇文献等），所以 A 是正式投稿版，B 是投稿前的稿子。

她那句话的另一种读法和证据一致：「nc-main（TexPage 上的 nc-main_submitted）是正式投稿版，比你本地那份新」。如果她指的是本地 `nc_paper/nc-main.tex`，上面 1–4 条都不支持。

09-08 的 REVISION_PLAN.md 写「submitted manuscript is nc-main.tex … confirmed against submitted.pdf」，当时只是为了区分旧的 AGU `main.tex`，没有逐句比。

## 她的决定

2026-11-27 14:30 用户按钮 GQ：「以TexPage投稿版为准」。tracked Word、一致性表、submitted_vs_draft.md 都以 A 为原稿，保持不变。
