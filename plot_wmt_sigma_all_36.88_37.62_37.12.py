#!/usr/bin/env python
"""
最终AABW分析脚本 (修正版)

修正内容:
1. evap, snow 已乘以开放水域覆盖率 (1-A)
2. WMT分解为: Heat + SeaIce + Other_FW
   - Other_FW = prec + evap + snow + runoff
3. AABW生成率 = WMT在阈值密度处的值

输出PDF图表
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from scipy import interpolate

EXPERIMENTS = ['pi', 'mh', 'lig', 'lgm', 'mis']
EXP_NAMES = {
    'pi': 'PI',
    'mh': 'MH',
    'lig': 'LIG',
    'lgm': 'LGM',
    'mis': 'MIS3'
}
EXP_COLORS = {
    'pi': '#2c3e50',
    'mh': '#27ae60',
    'lig': '#f39c12',
    'lgm': '#3498db',
    'mis': '#9b59b6'
}

REGIONS = {
    'southern_ocean': 'Southern Ocean',
    'weddell_sea': 'Weddell Sea',
    'ross_sea': 'Ross Sea',
    'adelie': 'Adélie Land'
}

AABW_THRESHOLDS = {
    'pi': 36.625,
    'mh': 36.625,
    'lig': 36.625,
    'lgm': 37.625,  # PI + 1.069
    'mis': 37.125    # PI + 0.68
}

OUTPUT_DIR = Path('./')
FIGURES_DIR = OUTPUT_DIR / 'figures'
DATA_DIR = OUTPUT_DIR / 'data'

OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

print('='*80)
print('AABW生成率最终分析 (修正版)')
print('='*80)
print('WMT分解: Heat + SeaIce + Other_FW')
print('='*80)
print()

# ============================================================================
# 函数
# ============================================================================

def load_wmt_data(exp, region):
    filename = f'wmt_echam_heat_fesom_freshwater_{region}_{exp}.nc'
    filepath = Path(f'{exp}') / filename
    if not filepath.exists():
        return None
    return xr.open_dataset(filepath)

def get_wmt_at_density(sigma2, wmt_values, target_sigma):
    f = interpolate.interp1d(sigma2, wmt_values, kind='linear',
                             bounds_error=False, fill_value=np.nan)
    return float(f(target_sigma))

def calculate_aabw_formation(ds, threshold_sigma2, months=None):
    """
    计算AABW形成率

    参数:
        ds: xarray Dataset
        threshold_sigma2: 阈值密度
        months: 要选择的月份列表 (1-12)，None表示全年
    """
    sigma2 = ds['sigma2_l_target'].values

    # 如果指定了月份，先筛选
    if months is not None:
        ds = ds.sel(time=ds['time.month'].isin(months))

    ds_mean = ds.mean(dim='time')

    results = {}

    # 热通量
    heat_vars = {
        'total_heat': 'total_heat_surface_exchange_flux_nonadvective_heat',
        'sw': 'sw_only_surface_exchange_flux_nonadvective_heat',
        'lw': 'lw_only_surface_exchange_flux_nonadvective_heat',
        'latent': 'latent_only_surface_exchange_flux_nonadvective_heat',
        'sensible': 'sensible_only_surface_exchange_flux_nonadvective_heat'
    }

    for key, var in heat_vars.items():
        if var in ds_mean:
            data = -ds_mean[var].values / 1e9
            results[f'heat_{key}'] = get_wmt_at_density(sigma2, data, threshold_sigma2)

    # 淡水通量 - 分为seaice和other_fw
    seaice_var = 'surface_ocean_flux_advective_negative_rhs_sea_ice_melt_salt'
    if seaice_var in ds_mean:
        seaice_data = -ds_mean[seaice_var].values / 1e9
        results['fw_seaice'] = get_wmt_at_density(sigma2, seaice_data, threshold_sigma2)

    # other_fw分量
    other_fw_vars = {
        'evap': 'surface_ocean_flux_advective_negative_rhs_evaporation_salt',
        'rain': 'surface_ocean_flux_advective_negative_rhs_rain_and_ice_salt',
        'snow': 'surface_ocean_flux_advective_negative_rhs_snow_salt',
        'river': 'surface_ocean_flux_advective_negative_rhs_rivers_salt'
    }

    other_fw_total = 0.0
    for key, var in other_fw_vars.items():
        if var in ds_mean:
            data = -ds_mean[var].values / 1e9
            val = get_wmt_at_density(sigma2, data, threshold_sigma2)
            results[f'fw_{key}'] = val
            other_fw_total += val

    results['fw_other'] = other_fw_total
    results['total_heat'] = results.get('heat_total_heat', 0)
    results['total_seaice'] = results.get('fw_seaice', 0)
    results['total_other_fw'] = other_fw_total
    results['total_wmt'] = results['total_heat'] + results['total_seaice'] + results['total_other_fw']

    return results

# ============================================================================
# 加载数据
# ============================================================================

print('加载数据并计算AABW生成率')
print('-'*80)

all_results = {}
all_results_winter = {}  # 6-9月数据

for exp in EXPERIMENTS:
    print(f'\n{EXP_NAMES[exp]}:')
    all_results[exp] = {}
    all_results_winter[exp] = {}
    threshold = AABW_THRESHOLDS[exp]

    for region_key, region_name in REGIONS.items():
        ds = load_wmt_data(exp, region_key)
        if ds is not None:
            # 全年平均
            results = calculate_aabw_formation(ds, threshold)
            all_results[exp][region_key] = results
            print(f'  {region_name:15s}: {results["total_wmt"]:+7.2f} Sv '
                  f'(Heat:{results["total_heat"]:+6.2f}, SeaIce:{results["total_seaice"]:+6.2f}, '
                  f'Other_FW:{results["total_other_fw"]:+6.2f})')

            # 6-9月平均
            results_winter = calculate_aabw_formation(ds, threshold, months=[6, 7, 8])
            all_results_winter[exp][region_key] = results_winter

            ds.close()

# ============================================================================
# 保存数据
# ============================================================================

print('\n' + '='*80)
print('保存数据')
print('-'*80)

data = []
for exp in EXPERIMENTS:
    for region in REGIONS.keys():
        if exp in all_results and region in all_results[exp]:
            res = all_results[exp][region]
            row = {
                'Experiment': EXP_NAMES[exp],
                'Region': REGIONS[region],
                'Threshold_σ2': AABW_THRESHOLDS[exp],
                'Total_WMT': res['total_wmt'],
                'Heat': res['total_heat'],
                'SeaIce': res['total_seaice'],
                'Other_FW': res['total_other_fw']
            }
            data.append(row)

df_results = pd.DataFrame(data)
csv_file = DATA_DIR / 'aabw_formation_rates_final.csv'
df_results.to_csv(csv_file, index=False, float_format='%.3f')
print(f'✓ {csv_file}')

# Anomalies
anomalies = []
for exp in ['mh', 'lig', 'lgm', 'mis']:
    for region in REGIONS.keys():
        if (exp in all_results and region in all_results[exp] and
            'pi' in all_results and region in all_results['pi']):
            exp_res = all_results[exp][region]
            pi_res = all_results['pi'][region]

            anom = {
                'Experiment': EXP_NAMES[exp],
                'Region': REGIONS[region],
                'ΔWMT': exp_res['total_wmt'] - pi_res['total_wmt'],
                'ΔHeat': exp_res['total_heat'] - pi_res['total_heat'],
                'ΔSeaIce': exp_res['total_seaice'] - pi_res['total_seaice'],
                'ΔOther_FW': exp_res['total_other_fw'] - pi_res['total_other_fw']
            }
            anomalies.append(anom)

df_anom = pd.DataFrame(anomalies)
anom_file = DATA_DIR / 'aabw_formation_anomalies_final.csv'
df_anom.to_csv(anom_file, index=False, float_format='%.3f')
print(f'✓ {anom_file}')

# ============================================================================
# 图1: AABW形成率对比
# ============================================================================

print('\n' + '='*80)
print('生成PDF图表')
print('-'*80)

print('图1: AABW形成率对比...')

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes_flat = axes.flatten()

for idx, (region_key, region_name) in enumerate(REGIONS.items()):
    ax = axes_flat[idx]

    wmt_values = []
    heat_values = []
    seaice_values = []
    other_fw_values = []
    labels = []

    for exp in EXPERIMENTS:
        if exp in all_results and region_key in all_results[exp]:
            res = all_results[exp][region_key]
            wmt_values.append(res['total_wmt'])
            heat_values.append(res['total_heat'])
            seaice_values.append(res['total_seaice'])
            other_fw_values.append(res['total_other_fw'])
            labels.append(EXP_NAMES[exp])

    x = np.arange(len(labels))
    width = 0.2

    bars1 = ax.bar(x - 1.5*width, wmt_values, width, label='Total WMT',
                   color='black', alpha=0.8)
    bars2 = ax.bar(x - 0.5*width, heat_values, width, label='Heat',
                   color='#e74c3c', alpha=0.8)
    bars3 = ax.bar(x + 0.5*width, seaice_values, width, label='Sea Ice',
                   color='#16a085', alpha=0.8)
    bars4 = ax.bar(x + 1.5*width, other_fw_values, width, label='Other FW',
                   color='#3498db', alpha=0.8)

    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=7)

    ax.set_xlabel('Experiment', fontweight='bold')
    ax.set_ylabel('AABW Formation Rate (Sv)', fontweight='bold')
    ax.set_title(region_name, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.8, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('AABW Formation Rate: Heat + SeaIce + Other FW',
             fontweight='bold', fontsize=16, y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])

fig1_file = FIGURES_DIR / 'plot_wmt_sigma_all_annual.pdf'
plt.savefig(fig1_file, bbox_inches='tight')
print(f'  ✓ {fig1_file.name}')
plt.close()

# ============================================================================
# 图2: AABW形成率对比 (6-9月平均)
# ============================================================================

print('图2: AABW形成率对比 (6-9月平均)...')

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes_flat = axes.flatten()

for idx, (region_key, region_name) in enumerate(REGIONS.items()):
    ax = axes_flat[idx]

    wmt_values = []
    heat_values = []
    seaice_values = []
    other_fw_values = []
    labels = []

    for exp in EXPERIMENTS:
        if exp in all_results_winter and region_key in all_results_winter[exp]:
            res = all_results_winter[exp][region_key]
            wmt_values.append(res['total_wmt'])
            heat_values.append(res['total_heat'])
            seaice_values.append(res['total_seaice'])
            other_fw_values.append(res['total_other_fw'])
            labels.append(EXP_NAMES[exp])

    x = np.arange(len(labels))
    width = 0.2

    bars1 = ax.bar(x - 1.5*width, wmt_values, width, label='Total WMT',
                   color='black', alpha=0.8)
    bars2 = ax.bar(x - 0.5*width, heat_values, width, label='Heat',
                   color='#e74c3c', alpha=0.8)
    bars3 = ax.bar(x + 0.5*width, seaice_values, width, label='Sea Ice',
                   color='#16a085', alpha=0.8)
    bars4 = ax.bar(x + 1.5*width, other_fw_values, width, label='Other FW',
                   color='#3498db', alpha=0.8)

    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=7)

    ax.set_xlabel('Experiment', fontweight='bold')
    ax.set_ylabel('AABW Formation Rate (Sv)', fontweight='bold')
    ax.set_title(region_name, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.8, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('AABW Formation Rate (Jun-August Average): Heat + SeaIce + Other FW',
             fontweight='bold', fontsize=16, y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])

fig2_file = FIGURES_DIR / 'plot_wmt_sigma_all_winter.pdf'
plt.savefig(fig2_file, bbox_inches='tight')
print(f'  ✓ {fig2_file.name}')
plt.close()

print('\n' + '='*80)
print('完成！')
print('='*80)
print(f'结果目录: {OUTPUT_DIR}/')
