# Brief: does the revised manuscript do what the reply letter says? (read-only review)

Paper: AWI-ESM Southern Ocean water mass transformation (WMT) across five climate states, Nature Communications
major revision. The author asked (11-27 13:38): check that every change the reply letter promises is really in the
revised manuscript, that it is really a change against the original submission, and the reverse.

Files (all in this folder: /tmp/claude-22370/-work-ba1066-a270064-cc-projects-aabw-5exps/d1f4281c-1793-4a52-ae53-a207b50475a1/scratchpad/review/):
- response_new.tex / response_new.pdf: the reply letter (current, compiled). Reviewer comments are \comment{Rk}{...}
  (black), replies start with \response (blue), quotes of the revised manuscript are in \begin{change}...\end{change}.
  Figures in the letter are numbered R1..R20 (they are letter figures, not manuscript figures).
  Lines starting with % are source comments, not part of the letter: ignore them.
- revised.tex / revised.pdf / revised.aux: the revised manuscript. Use revised.aux (\newlabel) to map labels to the
  numbers the reader sees: main figures 1-8, supplementary figures A1-A13, Table 1. The PDF has line numbers? (check)
- original.tex / nc-main_submitted.pdf / nc-main_submitted.aux: the ORIGINAL SUBMISSION (baseline). Reviewers' line
  numbers ("Line 102") refer to the original's line-numbered PDF.
- latexdiff.tex: latexdiff original.tex -> revised.tex (\DIFadd / \DIFdel), handy to see what changed.
- quote_check.txt: an automatic sentence-by-sentence check of every \begin{change} quote against revised.tex. The 11
  remaining MISSING lines are known notation-only differences (citet, sigma_2 written as σ\textsubscript{2}, ±, a
  user comment inside a quote); still look at each and say if any is a real mismatch.

## What to check for your part of the letter (each reviewer comment and its reply)
1. Every promise in the reply ("we have added / revised / removed / now state / moved / the figure now shows ..."):
   is it in revised.tex (give section and the first words, or label)? Is it really a change against original.tex,
   or was it already there? If the reply says something was removed, is it gone from revised.tex?
2. Every pointer: manuscript figure/table numbers (Fig. 1-8, A1-A13, Table 1), section names, "Methods", and
   line numbers of the ORIGINAL quoted by the reviewer. Does the number in the letter match revised.aux?
3. Every quote (\begin{change}): already machine-checked; spot-check that the quote sits where the reply says
   ("the Discussion now reads", "the abstract now reads", "Methods").
4. Numbers stated in the reply text (not in quotes) that should agree with the manuscript: do they?

## Output
Write <your part>_table.md in this folder (Chinese is fine):
| # | letter (reviewer, comment no., short promise) | revised.tex location | status | note |
status: 一致 (consistent) / 缺失 (promised but not in revised) / 不一致 (in revised but differs, or number/pointer wrong)
/ 原稿已有 (the "change" was already in the original).
Then two lists:
- 文字小修: pure wording/number/pointer mismatches with the exact fix (file, old string, new string), where the
  author's settled content decides which side is right. Prefer fixing the letter to match the manuscript unless the
  manuscript is clearly the one that is wrong.
- 需要作者定: anything that is a content choice (a promise not kept, a removed thing still present, a claim that
  is not supported).
Do NOT edit any file. Be exact: quote the strings. Keep the report compact; the table is the main product.
In your final message give: counts per status, the two lists (short), and the path of your table.
