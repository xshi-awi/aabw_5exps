# SOLVED

- **SAM 一节放哪**（R2 要移到 SI，R3 认为是最可推广的结果）：2026-09-09 用户定折中，图全部进 SI，正文加强文字并做 push/pull 新分析，回复信对两人区别表述。
- **Source Data 表**（09-08 时还没做）：已生成 `nc_paper/SUBMISSION/Source_Data.xlsx`，脚本 `make_source_data.py`。
- **spin-up 长度看似三篇文献互相矛盾**：误判。GRL Text S1 确认 PI 1,500 年、四个古气候实验各 1,000 年、取最后 100 年，与投稿版一致（`docs/model_equilibrium.md`）。
- **R3 建议用 GLORYS 算 WMT**：GLORYS 系列没有表面通量场，做不了。09-21 先据此婉拒；10-15 改为用 ORAS5 算出来正面回答，并写明 ORAS5 的淡水不含海冰结冰项（`docs/reanalysis_wmt_feasibility.md`）。
- **SAM 阈值在脚本、论文、任务描述里写法不一**：实际一直是 1.2σ，2026-05-19 统一了文档和默认参数，数据不用重算。
- **27 个脚本读不到数据**：硬编码的旧路径 `/work/ba0989/...` 在 2026-09-08 统一改到 `/work/ba1066/a270064/cc_projects/aabw_5exps`。
- **fwflux 合成图四角空白**：现役脚本换回 `composite_sam/fw_data/` 那版（规则网格 contourf），见 `composite_sam/CLAUDE.md`。
- **D6 推 TexPage 卡在 docx 闸门**：网页端 `response_letter_R12_1015.docx` 是用户 10-15 05:58 上传的带中文批注草稿，此后的本地提交已照做。闸门的阻断判定只报 2 段批注，另打印的 55 段差异是拿 R1 单独草稿比的；改和完整版比只有批注、D6 的 10→11 和旧草稿与定稿之差。用户 2026-11-08 10:01 批「推，批注不要了」，用 `--reviewed 25b3800a5e19` 推送，TexPage commit 3d2ad39，回读核对一致，网页 docx 原样保留。
