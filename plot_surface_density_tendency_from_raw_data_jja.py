#!/usr/bin/env python
"""
绘制JJA平均的表面密度趋势项差异图 (从原始数据计算)
- 图1: 热通量引起的密度趋势 (total, SW+LW, LH+SH)
- 图2: 淡水通量引起的密度趋势 (total, sea ice alone, other FW)
- 3行(变量) × 5列(PI, MH-PI, LIG-PI, LGM-PI, MIS-PI)
- 南极投影,50°S以南
- 第1列显示PI绝对值,第2-5列显示异常(paleo - PI)

数据来源：
- 热通量: echam_clim.nc (T63网格, W/m²)
- 淡水通量: *_reg.nc (1度网格, m/s)
- SST/SSS: sst_reg.nc, sss_reg.nc (1度网格)

密度趋势计算：
- 热通量 → 密度趋势: ∂σ/∂t = -(α/cp) × Q [kg/(m³·s)]
- 淡水通量 → 密度趋势: ∂σ/∂t = β × S × FW × ρ_water [kg/(m³·s)]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
import gsw
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 配置
# ============================================================================
EXPERIMENTS = {
    'mh': 'Mid-Holocene',
    'lig': 'Last Interglacial',
    'lgm': 'Last Glacial Maximum',
    'mis': 'MIS3'
}

# JJA月份索引 (0-based: 5,6,7 对应 6,7,8月)
JJA_MONTHS = [5, 6, 7]

# 常数
RHO_WATER = 1000.0  # kg/m³

# 单位换算: 密度趋势从 per second -> per month
# 1 month = 30 天 (约定俗成的"月"长度)
SEC_PER_MONTH = 30.0 * 86400.0  # = 2,592,000 s
UNIT_LABEL = 'kg m⁻³ month⁻¹'

print('='*80)
print('绘制JJA表面密度趋势异常图 (从原始数据计算)')
print('='*80)
print()

# ============================================================================
# 1. 加载数据
# ============================================================================
print('步骤 1: 加载数据...')
print('-'*80)

# 存储所有试验的数据
data = {}

for exp in ['pi'] + list(EXPERIMENTS.keys()):
    print(f'\n  加载 {exp.upper()}...')
    data[exp] = {}

    # ------------------------------------------------------------------------
    # 1.1 加载ECHAM热通量数据 (T63网格)
    # ------------------------------------------------------------------------
    echam_file = f'{exp}/echam_clim.nc'
    echam_ds = xr.open_dataset(echam_file)

    # 提取热通量 (单位: W/m²)
    # 注意：根据surface_tendencies.py，这些需要乘以-1
    data[exp]['var92'] = -echam_ds['var92'].values  # LW
    data[exp]['var95'] = -echam_ds['var95'].values  # SW
    data[exp]['var111'] = -echam_ds['var111'].values  # LH
    data[exp]['var120'] = -echam_ds['var120'].values  # SH

    # 保存ECHAM网格信息（只需要一次）
    if exp == 'pi':
        echam_lon = echam_ds['lon'].values
        echam_lat = echam_ds['lat'].values
        echam_time = echam_ds['time'].values
        print(f'    ECHAM网格: {len(echam_lon)} × {len(echam_lat)}')

    echam_ds.close()

    # ------------------------------------------------------------------------
    # 1.2 加载SST/SSS数据 (1度网格)
    # ------------------------------------------------------------------------
    sst_ds = xr.open_dataset(f'{exp}/sst_reg.nc')
    sss_ds = xr.open_dataset(f'{exp}/sss_reg.nc')

    data[exp]['sst'] = sst_ds['sst'].values  # (time, depth_coord, lat, lon)
    data[exp]['sss'] = sss_ds['sss'].values

    # 保存1度网格信息（只需要一次）
    if exp == 'pi':
        lon_1deg = sst_ds['lon'].values
        lat_1deg = sst_ds['lat'].values
        time_1deg = sst_ds['time'].values
        print(f'    1度网格: {len(lon_1deg)} × {len(lat_1deg)}')

    sst_ds.close()
    sss_ds.close()

    # ------------------------------------------------------------------------
    # 1.3 加载淡水通量数据 (1度网格, 单位: m/s)
    # ------------------------------------------------------------------------
    prec_ds = xr.open_dataset(f'{exp}/prec_reg.nc')
    evap_ds = xr.open_dataset(f'{exp}/evap_reg.nc')
    snow_ds = xr.open_dataset(f'{exp}/snow_reg.nc')
    runoff_ds = xr.open_dataset(f'{exp}/runoff_reg.nc')
    fw_ds = xr.open_dataset(f'{exp}/fw_reg.nc')

    data[exp]['prec'] = prec_ds['prec'].values  # m/s
    data[exp]['evap'] = evap_ds['evap'].values  # m/s
    data[exp]['snow'] = snow_ds['snow'].values  # m/s
    data[exp]['runoff'] = runoff_ds['runoff'].values  # m/s
    data[exp]['fw'] = fw_ds['fw'].values  # m/s

    prec_ds.close()
    evap_ds.close()
    snow_ds.close()
    runoff_ds.close()
    fw_ds.close()

    # ------------------------------------------------------------------------
    # 1.4 加载海冰数据
    # ------------------------------------------------------------------------
    a_ice_ds = xr.open_dataset(f'{exp}/a_ice_reg.nc')
    data[exp]['a_ice'] = a_ice_ds['a_ice'].values
    a_ice_ds.close()

    print(f'    ✓ 数据加载完成')

print('\n✓ 所有数据加载完成')
print()

# ============================================================================
# 2. 插值ECHAM热通量到1度网格
# ============================================================================
print('步骤 2: 插值ECHAM热通量到1度网格...')
print('-'*80)

# 创建1度网格
lon_2d, lat_2d = np.meshgrid(lon_1deg, lat_1deg)

# 准备ECHAM网格用于插值
echam_lon_converted = np.where(echam_lon > 180, echam_lon - 360, echam_lon)
echam_lon_2d, echam_lat_2d = np.meshgrid(echam_lon_converted, echam_lat)
echam_points = np.column_stack([echam_lon_2d.flatten(), echam_lat_2d.flatten()])
target_points = np.column_stack([lon_2d.flatten(), lat_2d.flatten()])

ntime = len(JJA_MONTHS)
nlat = len(lat_1deg)
nlon = len(lon_1deg)

for exp in ['pi'] + list(EXPERIMENTS.keys()):
    print(f'  插值 {exp.upper()}...')

    # 只插值JJA月份
    for var_name in ['var92', 'var95', 'var111', 'var120']:
        var_data = data[exp][var_name]  # (time, lat, lon)

        # 提取JJA月份
        var_jja = var_data[JJA_MONTHS, :, :]  # (3, lat, lon)

        # 插值每个时间步
        var_interp = np.zeros((3, nlat, nlon), dtype=np.float32)
        for t in range(3):
            var_flat = var_jja[t, :, :].flatten()
            var_interp[t, :, :] = griddata(echam_points, var_flat, target_points,
                                           method='linear').reshape(nlat, nlon)

        # 替换原数据为插值后的JJA数据
        data[exp][var_name] = var_interp

    print(f'    ✓ 插值完成')

print('✓ ECHAM数据插值完成')
print()

# ============================================================================
# 3. 提取JJA平均并计算密度趋势
# ============================================================================
print('步骤 3: 计算JJA平均并转换为密度趋势...')
print('-'*80)

# 存储密度趋势数据
jja_density_tendency = {}

for exp in ['pi'] + list(EXPERIMENTS.keys()):
    print(f'\n  处理 {exp.upper()}...')

    jja_density_tendency[exp] = {}

    # ------------------------------------------------------------------------
    # 3.1 提取JJA平均的SST和SSS (去除depth_coord维度)
    # ------------------------------------------------------------------------
    sst_all = data[exp]['sst']  # (time, depth_coord, lat, lon)
    sss_all = data[exp]['sss']

    # 去除depth_coord维度并提取JJA
    sst_jja = sst_all[JJA_MONTHS, 0, :, :].mean(axis=0)  # (lat, lon)
    sss_jja = sss_all[JJA_MONTHS, 0, :, :].mean(axis=0)

    # ------------------------------------------------------------------------
    # 3.2 计算GSW参数 (α, β, cp)
    # ------------------------------------------------------------------------
    print(f'    计算GSW参数...')

    SA = gsw.SA_from_SP(sss_jja, p=0, lon=lon_2d, lat=lat_2d)
    CT = gsw.CT_from_t(SA, sst_jja, p=0)

    alpha = gsw.alpha(SA, CT, p=0)  # thermal expansion coefficient [1/K]
    beta = gsw.beta(SA, CT, p=0)    # haline contraction coefficient [kg/m³/(g/kg)]
    cp = gsw.cp_t_exact(SA, sst_jja, p=0)  # specific heat capacity [J/(kg·K)]

    print(f'      Alpha 范围: [{np.nanmin(alpha):.6f}, {np.nanmax(alpha):.6f}] K⁻¹')
    print(f'      Beta 范围: [{np.nanmin(beta):.6f}, {np.nanmax(beta):.6f}]')
    print(f'      Cp 范围: [{np.nanmin(cp):.1f}, {np.nanmax(cp):.1f}] J/(kg·K)')

    # ------------------------------------------------------------------------
    # 3.3 热通量 → 密度趋势
    # ------------------------------------------------------------------------
    # 公式: ∂σ/∂t = -(α/cp) × Q
    # Q: 热通量 [W/m²]
    # 输出: 密度趋势 [kg/(m³·s)]

    # 提取JJA平均热通量 (已经插值到1度网格)
    heat_lw = data[exp]['var92'].mean(axis=0)  # (lat, lon), W/m²
    heat_sw = data[exp]['var95'].mean(axis=0)
    heat_lh = data[exp]['var111'].mean(axis=0)
    heat_sh = data[exp]['var120'].mean(axis=0)

    # 计算热通量引起的密度趋势 [kg/(m³·s)]
    jja_density_tendency[exp]['heat_lw'] = -(alpha / cp) * heat_lw
    jja_density_tendency[exp]['heat_sw'] = -(alpha / cp) * heat_sw
    jja_density_tendency[exp]['heat_lh'] = -(alpha / cp) * heat_lh
    jja_density_tendency[exp]['heat_sh'] = -(alpha / cp) * heat_sh

    jja_density_tendency[exp]['heat_total'] = (
        jja_density_tendency[exp]['heat_lw'] +
        jja_density_tendency[exp]['heat_sw'] +
        jja_density_tendency[exp]['heat_lh'] +
        jja_density_tendency[exp]['heat_sh']
    )

    jja_density_tendency[exp]['heat_radiation'] = (
        jja_density_tendency[exp]['heat_sw'] +
        jja_density_tendency[exp]['heat_lw']
    )

    jja_density_tendency[exp]['heat_turbulent'] = (
        jja_density_tendency[exp]['heat_lh'] +
        jja_density_tendency[exp]['heat_sh']
    )

    # 反转热通量符号(用于绘图)
    jja_density_tendency[exp]['heat_total'] = -jja_density_tendency[exp]['heat_total']
    jja_density_tendency[exp]['heat_radiation'] = -jja_density_tendency[exp]['heat_radiation']
    jja_density_tendency[exp]['heat_turbulent'] = -jja_density_tendency[exp]['heat_turbulent']

    # ------------------------------------------------------------------------
    # 3.4 淡水通量 → 密度趋势
    # ------------------------------------------------------------------------
    # 公式: ∂σ/∂t = β × S × FW × ρ_water
    # FW: 淡水通量 [m/s]
    # S: 盐度 [PSU]
    # 输出: 密度趋势 [kg/(m³·s)]

    # 提取JJA平均淡水通量 (depth_coord=0)
    prec_all = data[exp]['prec'][JJA_MONTHS, 0, :, :]  # (3, lat, lon)
    evap_all = data[exp]['evap'][JJA_MONTHS, 0, :, :]
    snow_all = data[exp]['snow'][JJA_MONTHS, 0, :, :]
    runoff_all = data[exp]['runoff'][JJA_MONTHS, 0, :, :]
    fw_all = data[exp]['fw'][JJA_MONTHS, 0, :, :]
    a_ice_all = data[exp]['a_ice'][JJA_MONTHS, 0, :, :]  # Also need depth_coord=0

    # JJA平均
    prec_jja = prec_all.mean(axis=0)  # m/s
    evap_jja = evap_all.mean(axis=0)
    snow_jja = snow_all.mean(axis=0)
    runoff_jja = runoff_all.mean(axis=0)
    fw_jja = -fw_all.mean(axis=0)  # 注意负号
    a_ice_jja = a_ice_all.mean(axis=0)

    # 将runoff的NaN值替换为0
    runoff_jja = np.nan_to_num(runoff_jja, nan=0.0)

    # 不使用开放水域覆盖率修正,直接使用原始通量
    prec_eff = prec_jja
    snow_eff = snow_jja
    evap_eff = evap_jja
    runoff_eff = runoff_jja

    # 海冰通量 = fw总通量 - 其他分量
    seaice_eff = fw_jja - (prec_eff + snow_eff + evap_eff + runoff_eff)

    # 应用海冰mask
    ice_mask = a_ice_jja > 0
    seaice_eff = np.where(ice_mask, seaice_eff, 0)

    # 计算淡水通量引起的密度趋势 [kg/(m³·s)]
    # 注意: 淡水通量单位是 m/s，需要乘以 ρ_water 转换为 kg/(m²·s)
    # ∂σ/∂t = β × S × (FW [m/s] × ρ_water [kg/m³]) / ρ_water
    #        = β × S × FW [m/s]

    # 但根据surface_tendencies.py第326-330行:
    # sigma_tendency = -RHO_REF * beta * sss * fw_flux / RHO_WATER
    # 其中 fw_flux 的单位是 kg/(m²·s)
    # 所以我们需要: FW [m/s] × RHO_WATER → kg/(m²·s)

    # 最终公式: ∂σ/∂t = -RHO_REF * beta * sss * (FW [m/s] * RHO_WATER) / RHO_WATER
    #                   = -RHO_REF * beta * sss * FW [m/s]

    # 但实际上，根据物理意义:
    # 淡水输入会稀释盐度，降低密度
    # 所以应该是负号

    # 让我重新理解：
    # fw, prec, snow, runoff 是正向海洋的通量（增加质量）
    # evap 是负向海洋的通量（减少质量）
    # 所以 evap < 0

    # 密度通量 = β × S × FW_mass_flux
    # FW_mass_flux = FW [m/s] × ρ_water [kg/m³] → [kg/(m²·s)]

    jja_density_tendency[exp]['mass_prec'] = -beta * sss_jja * prec_eff * RHO_WATER / RHO_WATER
    jja_density_tendency[exp]['mass_evap'] = -beta * sss_jja * evap_eff * RHO_WATER / RHO_WATER
    jja_density_tendency[exp]['mass_snow'] = -beta * sss_jja * snow_eff * RHO_WATER / RHO_WATER
    jja_density_tendency[exp]['mass_runoff'] = -beta * sss_jja * runoff_eff * RHO_WATER / RHO_WATER
    jja_density_tendency[exp]['mass_seaice'] = -beta * sss_jja * seaice_eff * RHO_WATER / RHO_WATER

    # 简化: RHO_WATER 约掉
    jja_density_tendency[exp]['mass_prec'] = -beta * sss_jja * prec_eff
    jja_density_tendency[exp]['mass_evap'] = -beta * sss_jja * evap_eff
    jja_density_tendency[exp]['mass_snow'] = -beta * sss_jja * snow_eff
    jja_density_tendency[exp]['mass_runoff'] = -beta * sss_jja * runoff_eff
    jja_density_tendency[exp]['mass_seaice'] = -beta * sss_jja * seaice_eff

    jja_density_tendency[exp]['mass_total'] = (
        jja_density_tendency[exp]['mass_prec'] +
        jja_density_tendency[exp]['mass_evap'] +
        jja_density_tendency[exp]['mass_snow'] +
        jja_density_tendency[exp]['mass_runoff'] +
        jja_density_tendency[exp]['mass_seaice']
    )

    jja_density_tendency[exp]['mass_other'] = (
        jja_density_tendency[exp]['mass_prec'] +
        jja_density_tendency[exp]['mass_evap'] +
        jja_density_tendency[exp]['mass_snow'] +
        jja_density_tendency[exp]['mass_runoff']
    )

    # 使用fw本身的mask: fw为NaN的地方,所有变量也设为NaN
    # 这样可以避免LGM/MIS在Ross Sea shelf等陆地区域显示sea ice值
    fw_mask = np.isnan(fw_jja)
    for var_key in ['heat_total', 'heat_radiation', 'heat_turbulent',
                    'mass_total', 'mass_seaice', 'mass_other']:
        jja_density_tendency[exp][var_key] = np.where(fw_mask, np.nan, jja_density_tendency[exp][var_key])

    print(f'    ✓ 密度趋势计算完成')

print('\n✓ 所有试验的密度趋势计算完成')
print()

# ============================================================================
# 4. 预处理: 平滑数据、添加循环点
# ============================================================================
print('步骤 4: 预处理数据平滑、循环点和网格...')
print('-'*80)

def smooth_data(data, sigma=1.5):
    """Apply Gaussian smoothing to data, handling NaN values"""
    data_filled = np.nan_to_num(data, nan=0.0)
    mask = ~np.isnan(data)
    mask_float = mask.astype(float)

    smoothed = gaussian_filter(data_filled, sigma=sigma)
    smoothed_mask = gaussian_filter(mask_float, sigma=sigma)

    with np.errstate(divide='ignore', invalid='ignore'):
        result = smoothed / smoothed_mask
        result[smoothed_mask < 0.1] = np.nan

    return result

def add_cyclic_point(data, lon):
    """添加循环点避免经度缝隙"""
    if len(data.shape) == 2:
        cyclic_data = np.concatenate([data, data[:, 0:1]], axis=1)
    else:
        cyclic_data = np.concatenate([data, data[0:1]], axis=0)
    cyclic_lon = np.concatenate([lon, [lon[0] + 360]])
    return cyclic_data, cyclic_lon

# 预处理所有数据
jja_cyclic = {}
for exp in ['pi'] + list(EXPERIMENTS.keys()):
    jja_cyclic[exp] = {}

    for var in ['heat_total', 'heat_radiation', 'heat_turbulent',
                'mass_total', 'mass_seaice', 'mass_other']:
        # 单位换算: per second -> per month
        data_vals = jja_density_tendency[exp][var] * SEC_PER_MONTH

        # 平滑数据 (不再在这里应用ice_mask,因为fw_mask已经处理了)
        data_smooth = smooth_data(data_vals, sigma=1.5)

        # 添加循环点
        jja_cyclic[exp][var], lon_cyclic = add_cyclic_point(data_smooth, lon_1deg)

# 创建循环mesh grid
lon_2d_cyclic, lat_2d_cyclic = np.meshgrid(lon_cyclic, lat_1deg)

print('✓ 预处理完成')
print()

# ============================================================================
# 5. 预计算所有异常并确定colorbar范围
# ============================================================================
print('步骤 5: 预计算异常并确定colorbar范围...')
print('-'*80)

# 只统计50°S以南区域
# 注意: 需要创建与lon_2d_cyclic形状匹配的mask
# lon_2d_cyclic 和 lat_2d_cyclic 的形状是 (180, 361)
mask_so = lat_2d_cyclic < -50

anomalies = {}
for exp in EXPERIMENTS.keys():
    anomalies[exp] = {}
    for var in ['heat_total', 'heat_radiation', 'heat_turbulent',
                'mass_total', 'mass_seaice', 'mass_other']:
        anomalies[exp][var] = jja_cyclic[exp][var] - jja_cyclic['pi'][var]

# 确定colorbar范围
heat_vars = ['heat_total', 'heat_radiation', 'heat_turbulent']
mass_vars = ['mass_total', 'mass_seaice', 'mass_other']

# ============================================================================
# 策略: 每张图内部统一colorbar范围
# - 图1 (热通量): 所有PI列统一范围, 所有异常列统一范围
# - 图2 (淡水通量): 所有PI列统一范围, 所有异常列统一范围
# ============================================================================

# 图1 (热通量): 收集所有3个变量的PI和异常值
print('\n图1 (热通量) colorbar范围:')
heat_pi_all_vals = []
heat_anom_all_vals = []

for var in heat_vars:
    # PI值
    pi_vals = jja_cyclic['pi'][var][mask_so]
    pi_vals = pi_vals[~np.isnan(pi_vals)]
    heat_pi_all_vals.extend(pi_vals)

    # 异常值
    for exp in EXPERIMENTS.keys():
        anom_vals = anomalies[exp][var][mask_so]
        anom_vals = anom_vals[~np.isnan(anom_vals)]
        heat_anom_all_vals.extend(anom_vals)

# 计算图1的统一范围
heat_pi_vmin, heat_pi_vmax = np.percentile(heat_pi_all_vals, [1, 99])
heat_pi_vmax_abs = max(abs(heat_pi_vmin), abs(heat_pi_vmax))

heat_anom_vmax_abs = np.percentile(np.abs(heat_anom_all_vals), 99)

print(f'  PI列统一范围: [{-heat_pi_vmax_abs:.2e}, {heat_pi_vmax_abs:.2e}] {UNIT_LABEL}')
print(f'  异常列统一范围: ±{heat_anom_vmax_abs:.2e} {UNIT_LABEL}')

# 图2 (淡水通量): 收集所有3个变量的PI和异常值
print('\n图2 (淡水通量) colorbar范围:')
mass_pi_all_vals = []
mass_anom_all_vals = []

for var in mass_vars:
    # PI值
    pi_vals = jja_cyclic['pi'][var][mask_so]
    pi_vals = pi_vals[~np.isnan(pi_vals)]
    mass_pi_all_vals.extend(pi_vals)

    # 异常值
    for exp in EXPERIMENTS.keys():
        anom_vals = anomalies[exp][var][mask_so]
        anom_vals = anom_vals[~np.isnan(anom_vals)]
        mass_anom_all_vals.extend(anom_vals)

# 计算图2的统一范围
mass_pi_vmin, mass_pi_vmax = np.percentile(mass_pi_all_vals, [1, 99])
mass_pi_vmax_abs = max(abs(mass_pi_vmin), abs(mass_pi_vmax))

mass_anom_vmax_abs = np.percentile(np.abs(mass_anom_all_vals), 99)

print(f'  PI列统一范围: [{-mass_pi_vmax_abs:.2e}, {mass_pi_vmax_abs:.2e}] {UNIT_LABEL}')
print(f'  异常列统一范围: ±{mass_anom_vmax_abs:.2e} {UNIT_LABEL}')

print('\n✓ 异常计算完成')
print()

# ============================================================================
# 6. 绘图函数
# ============================================================================

def get_nice_ticks(vmax, n_ticks=5):
    """生成对称的等间距刻度"""
    import math

    magnitude = 10 ** math.floor(math.log10(vmax))
    nice_numbers = [1, 2, 2.5, 5, 10]

    for nice in nice_numbers:
        candidate = nice * magnitude
        if candidate >= vmax:
            vmax_nice = candidate
            break
    else:
        vmax_nice = 10 * magnitude

    ticks = np.linspace(-vmax_nice, vmax_nice, n_ticks)

    return ticks.tolist(), vmax_nice

def pick_tick_format(vmax_nice):
    """根据数值量级选择colorbar刻度格式 (per-month值通常为O(0.01)-O(10))"""
    if vmax_nice >= 100 or vmax_nice < 0.01:
        return '%.1e'
    elif vmax_nice >= 10:
        return '%.0f'
    elif vmax_nice >= 1:
        return '%.1f'
    elif vmax_nice >= 0.1:
        return '%.2f'
    else:
        return '%.3f'

def plot_field(ax, lon_2d, lat_2d, data, vmin, vmax, cmap, panel_label):
    """绘制场"""
    levels = np.linspace(vmin, vmax, 40)

    cf = ax.contourf(lon_2d, lat_2d, data,
                     levels=levels,
                     cmap=cmap, extend='both',
                     transform=ccrs.PlateCarree(),
                     zorder=1)

    # 陆地要在数据之上,使用zorder=2
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=2)
    ax.coastlines(resolution='110m', linewidth=0.5, color='black', zorder=3)

    ax.text(0.02, 0.98, panel_label, transform=ax.transAxes,
            fontsize=15, weight='bold', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85),
            zorder=4)

    ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.3, linestyle='--')

    return cf

# ============================================================================
# 7. 图1: 热通量引起的密度趋势
# ============================================================================
print('步骤 6: 绘制图1 - 热通量引起的密度趋势...')
print('-'*80)

fig = plt.figure(figsize=(18, 15))
gs = GridSpec(3, 5, figure=fig, hspace=0.55, wspace=0.08,
              left=0.05, right=0.95, top=0.96, bottom=0.05)

var_keys = ['heat_total', 'heat_radiation', 'heat_turbulent']
var_titles = ['Total Heat', 'SW + LW (Radiation)', 'LH + SH (Turbulent)']
# 第一列(PI)colorbar用短标题,去掉括号说明避免单列colorbar太窄被截断
var_titles_pi = ['Total Heat', 'SW + LW', 'LH + SH']

exp_list = list(EXPERIMENTS.keys())

# 使用统一的colorbar范围
ticks_pi, vmax_pi_nice = get_nice_ticks(heat_pi_vmax_abs, n_ticks=5)
ticks_anom, vmax_anom_nice = get_nice_ticks(heat_anom_vmax_abs, n_ticks=5)
fmt_pi = pick_tick_format(vmax_pi_nice)
fmt_anom = pick_tick_format(vmax_anom_nice)

panel_idx = 0

for row, (var_key, var_title, var_title_pi) in enumerate(zip(var_keys, var_titles, var_titles_pi)):
    print(f'  绘制第{row+1}行: {var_title}...')

    all_cfs = []
    all_axes = []

    # 第1列: PI (使用统一的PI范围)
    ax = fig.add_subplot(gs[row, 0], projection=ccrs.SouthPolarStereo())
    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    all_axes.append(ax)

    panel_label = f'({chr(97+panel_idx)})'
    panel_idx += 1

    cf_pi = plot_field(ax, lon_2d_cyclic, lat_2d_cyclic, jja_cyclic['pi'][var_key],
                       -vmax_pi_nice, vmax_pi_nice, 'RdBu_r', panel_label)
    all_cfs.append(cf_pi)
    # climate-state column headings on the top row (Reviewer 1, comment 17)
    if row == 0:
        ax.set_title('PI', fontsize=19, fontweight='bold', pad=10)

    # 第2-5列: 异常 (使用统一的异常范围)
    for col, exp in enumerate(exp_list, start=1):
        ax = fig.add_subplot(gs[row, col], projection=ccrs.SouthPolarStereo())
        ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
        all_axes.append(ax)

        panel_label = f'({chr(97+panel_idx)})'
        panel_idx += 1

        cf_anom = plot_field(ax, lon_2d_cyclic, lat_2d_cyclic, anomalies[exp][var_key],
                            -vmax_anom_nice, vmax_anom_nice, 'RdBu_r', panel_label)
        all_cfs.append(cf_anom)
        if row == 0:
            EXPNAME = {'mh': 'MH', 'lig': 'LIG', 'lgm': 'LGM', 'mis': 'MIS3'}
            ax.set_title(EXPNAME.get(exp, exp.upper()) + ' $-$ PI',
                         fontsize=19, fontweight='bold', pad=10)

    # Colorbars
    pos0 = all_axes[0].get_position()
    cbar_ax_pi = fig.add_axes([pos0.x0, pos0.y0 - 0.055, pos0.width, 0.018])
    cbar_pi = plt.colorbar(all_cfs[0], cax=cbar_ax_pi, orientation='horizontal',
                           ticks=ticks_pi, format=fmt_pi)
    cbar_pi.set_label(f'PI: {var_title_pi} ({UNIT_LABEL})', fontsize=13)
    cbar_pi.ax.tick_params(labelsize=14)

    pos1 = all_axes[1].get_position()
    pos4 = all_axes[4].get_position()
    cbar_ax_anom = fig.add_axes([pos1.x0, pos1.y0 - 0.055, pos4.x1 - pos1.x0, 0.018])
    cbar_anom = plt.colorbar(all_cfs[1], cax=cbar_ax_anom, orientation='horizontal',
                             ticks=ticks_anom, format=fmt_anom)
    cbar_anom.set_label(f'Anomaly: {var_title} ({UNIT_LABEL})', fontsize=16)
    cbar_anom.ax.tick_params(labelsize=14)

output_file1 = 'figures/surface_heat_density_tendency_from_raw_jja.pdf'
plt.savefig(output_file1, dpi=200, bbox_inches='tight')
print(f'✓ 图1保存: {output_file1}')
plt.close()

# ============================================================================
# 8. 图2: 淡水通量引起的密度趋势
# ============================================================================
print('\n步骤 7: 绘制图2 - 淡水通量引起的密度趋势...')
print('-'*80)

fig = plt.figure(figsize=(18, 15))
gs = GridSpec(3, 5, figure=fig, hspace=0.55, wspace=0.08,
              left=0.05, right=0.95, top=0.96, bottom=0.05)

var_keys = ['mass_total', 'mass_seaice', 'mass_other']
var_titles = ['Total Freshwater', 'Sea Ice Alone', 'Other FW (net precipitation + runoff)']
# 第一列(PI)colorbar用短标题,避免最下面那个括号太长被截断
var_titles_pi = ['Total Freshwater', 'Sea Ice Alone', 'Other FW']

# 使用统一的colorbar范围
ticks_pi, vmax_pi_nice = get_nice_ticks(mass_pi_vmax_abs, n_ticks=5)
ticks_anom, vmax_anom_nice = get_nice_ticks(mass_anom_vmax_abs, n_ticks=5)
fmt_pi = pick_tick_format(vmax_pi_nice)
fmt_anom = pick_tick_format(vmax_anom_nice)

panel_idx = 0

for row, (var_key, var_title, var_title_pi) in enumerate(zip(var_keys, var_titles, var_titles_pi)):
    print(f'  绘制第{row+1}行: {var_title}...')

    all_cfs = []
    all_axes = []

    # 第1列: PI (使用统一的PI范围)
    ax = fig.add_subplot(gs[row, 0], projection=ccrs.SouthPolarStereo())
    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    all_axes.append(ax)

    panel_label = f'({chr(97+panel_idx)})'
    panel_idx += 1

    cf_pi = plot_field(ax, lon_2d_cyclic, lat_2d_cyclic, jja_cyclic['pi'][var_key],
                       -vmax_pi_nice, vmax_pi_nice, 'RdBu_r', panel_label)
    all_cfs.append(cf_pi)
    # climate-state column headings on the top row (Reviewer 1, comment 17)
    if row == 0:
        ax.set_title('PI', fontsize=19, fontweight='bold', pad=10)

    # 第2-5列: 异常 (使用统一的异常范围)
    for col, exp in enumerate(exp_list, start=1):
        ax = fig.add_subplot(gs[row, col], projection=ccrs.SouthPolarStereo())
        ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
        all_axes.append(ax)

        panel_label = f'({chr(97+panel_idx)})'
        panel_idx += 1

        cf_anom = plot_field(ax, lon_2d_cyclic, lat_2d_cyclic, anomalies[exp][var_key],
                            -vmax_anom_nice, vmax_anom_nice, 'RdBu_r', panel_label)
        all_cfs.append(cf_anom)
        if row == 0:
            EXPNAME = {'mh': 'MH', 'lig': 'LIG', 'lgm': 'LGM', 'mis': 'MIS3'}
            ax.set_title(EXPNAME.get(exp, exp.upper()) + ' $-$ PI',
                         fontsize=19, fontweight='bold', pad=10)

    # Colorbars
    pos0 = all_axes[0].get_position()
    cbar_ax_pi = fig.add_axes([pos0.x0, pos0.y0 - 0.055, pos0.width, 0.018])
    cbar_pi = plt.colorbar(all_cfs[0], cax=cbar_ax_pi, orientation='horizontal',
                           ticks=ticks_pi, format=fmt_pi)
    cbar_pi.set_label(f'PI: {var_title_pi} ({UNIT_LABEL})', fontsize=13)
    cbar_pi.ax.tick_params(labelsize=14)

    pos1 = all_axes[1].get_position()
    pos4 = all_axes[4].get_position()
    cbar_ax_anom = fig.add_axes([pos1.x0, pos1.y0 - 0.055, pos4.x1 - pos1.x0, 0.018])
    cbar_anom = plt.colorbar(all_cfs[1], cax=cbar_ax_anom, orientation='horizontal',
                             ticks=ticks_anom, format=fmt_anom)
    cbar_anom.set_label(f'Anomaly: {var_title} ({UNIT_LABEL})', fontsize=16)
    cbar_anom.ax.tick_params(labelsize=14)

output_file2 = 'figures/surface_freshwater_density_tendency_from_raw_jja.pdf'
plt.savefig(output_file2, dpi=200, bbox_inches='tight')
print(f'✓ 图2保存: {output_file2}')
plt.close()

# ============================================================================
# 完成
# ============================================================================
print('\n' + '='*80)
print('绘图完成!')
print('='*80)
print()
print('输出文件:')
print(f'  1. {output_file1}')
print(f'  2. {output_file2}')
print()

# 统计信息
print('密度趋势异常值统计 (Paleo - PI, 50°S以南区域平均):')
print('-'*80)

for exp, exp_name in EXPERIMENTS.items():
    print(f'\n{exp_name}:')

    heat_total_mean = np.nanmean(anomalies[exp]['heat_total'][mask_so])
    heat_rad_mean = np.nanmean(anomalies[exp]['heat_radiation'][mask_so])
    heat_turb_mean = np.nanmean(anomalies[exp]['heat_turbulent'][mask_so])

    mass_total_mean = np.nanmean(anomalies[exp]['mass_total'][mask_so])
    mass_ice_mean = np.nanmean(anomalies[exp]['mass_seaice'][mask_so])
    mass_other_mean = np.nanmean(anomalies[exp]['mass_other'][mask_so])

    print(f'  Heat Total:      {heat_total_mean:+.3f} {UNIT_LABEL}')
    print(f'  Heat Radiation:  {heat_rad_mean:+.3f} {UNIT_LABEL}')
    print(f'  Heat Turbulent:  {heat_turb_mean:+.3f} {UNIT_LABEL}')
    print(f'  Mass Total:      {mass_total_mean:+.3f} {UNIT_LABEL}')
    print(f'  Mass Sea Ice:    {mass_ice_mean:+.3f} {UNIT_LABEL}')
    print(f'  Mass Other:      {mass_other_mean:+.3f} {UNIT_LABEL}')
print()
