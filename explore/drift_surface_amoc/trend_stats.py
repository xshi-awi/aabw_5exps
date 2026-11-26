#!/usr/bin/env python
"""Linear trends of the yearly SST and overturning series, with autocorrelation-adjusted errors.

Trend: OLS slope against model year (LIG has gaps at 2318 and 2372; the real years are used).
Error: OLS standard error inflated with the lag-1 autocorrelation r1 of the residuals
(effective sample size n_eff = n (1 - r1) / (1 + r1), Santer et al. 2000, JGR 105, 7337;
r1 from year pairs one year apart; r1 <= 0 gives n_eff = n). 95% interval with
t(n_eff - 2). Also the SD of the annual means (raw, ddof=1) and of the detrended values.
Inter-state comparison: 100-year means and their differences from PI.
Writes trends.csv, state_means.csv and numbers.txt.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
EXPS = {"pi": "PI", "mh": "MH", "lig": "LIG", "lgm": "LGM", "mis": "MIS3"}
SERIES = {  # name: (file stem, column, unit, label)
    "sst_global": ("surface_yearly", "sst_global", "K", "global-mean SST"),
    "sst_so50": ("surface_yearly", "sst_so50", "K", "SST south of 50S"),
    "atl_upper": ("moc_indices_yearly", "atl_upper", "Sv", "Atlantic upper cell, max 500-4000 m, 20-70N"),
    "atl_amoc_ncl": ("moc_indices_yearly", "atl_amoc_ncl", "Sv", "Atlantic AMOC_NCL, max 800-2000 m, 20-90N"),
    "glob_upper": ("moc_indices_yearly", "glob_upper", "Sv", "global upper cell (Methods), max 500-4000 m, 20-70N"),
    "glob_abyssal": ("moc_indices_yearly", "glob_abyssal", "Sv", "global abyssal cell (Methods), min below 2000 m, south of 30S"),
}


def trend(t, y):
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    n = y.size
    res = stats.linregress(t, y)
    resid = y - (res.intercept + res.slope * t)
    order = np.argsort(t)
    ts, rs = t[order], resid[order]
    pair = np.diff(ts) == 1
    r1 = np.corrcoef(rs[:-1][pair], rs[1:][pair])[0, 1]
    neff = n * (1 - r1) / (1 + r1) if r1 > 0 else float(n)
    neff = min(neff, n)
    se = res.stderr * np.sqrt((n - 2) / (neff - 2))
    tcrit = stats.t.ppf(0.975, neff - 2)
    p = 2 * stats.t.sf(abs(res.slope / se), neff - 2)
    return dict(n=n, slope_per_century=100 * res.slope, se_ols=100 * res.stderr,
                r1=r1, n_eff=neff, se_adj=100 * se, ci95=100 * tcrit * se, p_adj=p,
                sd_annual=np.std(y, ddof=1), sd_detrended=np.std(resid, ddof=2), mean=y.mean())


def main():
    rows = []
    for exp, lab in EXPS.items():
        for name, (stem, col, unit, desc) in SERIES.items():
            d = pd.read_csv(HERE / f"{stem}_{exp}.csv")
            r = trend(d["year"].values, d[col].values)
            rows.append({"exp": lab, "series": name, "unit": unit,
                         "years": f"{d.year.min()}-{d.year.max()}", **r})
    tr = pd.DataFrame(rows)
    tr.to_csv(HERE / "trends.csv", index=False, float_format="%.4g")

    means = tr.pivot(index="series", columns="exp", values="mean")[list(EXPS.values())]
    means.to_csv(HERE / "state_means.csv", float_format="%.4f")

    lines = ["Trends over each state's 100-year analysis window (PI/MH/LIG 2300-2399, LGM 2400-2499,",
             "MIS3 2500-2599; LIG lacks 2318 and 2372). Slope per century +- SE adjusted for lag-1",
             "autocorrelation (n_eff); 95% CI half-width; SD of annual means; 100-yr mean; mean minus PI.", ""]
    for name, (stem, col, unit, desc) in SERIES.items():
        lines.append(f"== {name}: {desc} [{unit}]")
        lines.append(f"{'exp':5s} {'trend/cent':>10s} {'+-SEadj':>8s} {'CI95':>7s} {'p':>6s} "
                     f"{'r1':>5s} {'neff':>5s} {'SDann':>6s} {'SDdetr':>6s} {'mean':>8s} {'-PI':>7s} "
                     f"{'|trend|/|dPI|':>12s}")
        pi_mean = means.loc[name, "PI"]
        for lab in EXPS.values():
            r = tr[(tr.exp == lab) & (tr.series == name)].iloc[0]
            dpi = r["mean"] - pi_mean
            ratio = abs(r.slope_per_century) / abs(dpi) if lab != "PI" else np.nan
            lines.append(f"{lab:5s} {r.slope_per_century:10.3f} {r.se_adj:8.3f} {r.ci95:7.3f} "
                         f"{r.p_adj:6.3f} {r.r1:5.2f} {r.n_eff:5.1f} {r.sd_annual:6.3f} "
                         f"{r.sd_detrended:6.3f} {r['mean']:8.3f} {dpi:7.3f} {ratio:12.3f}")
        lines.append("")
    txt = "\n".join(lines)
    (HERE / "numbers.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
