#!/usr/bin/env python
"""
绘制JJA平均的表面密度趋势项差异图 (优化版)
- 图1: 热通量预算 (total, SW+LW, LH+SH)
- 图2: 淡水通量预算 (total, sea ice alone, other FW)
- 3行(变量) × 5列(PI, MH-PI, LIG-PI, LGM-PI, MIS-PI)
- 南极投影,50°S以南
- 第1列显示PI绝对值,第2-5列显示异常(paleo - PI)

性能优化:
- 预计算所有异常数据
- 预处理循环点
- 使用pcolormesh替代contourf (更快)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter
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

# 等值线数量 (减少以提高速度)
CONTOUR_LEVELS_HEAT = 8
CONTOUR_LEVELS_FW = 6

print('='*80)
print('绘制JJA表面密度趋势项异常图 (优化版)')
print('='*80)
print()

# ============================================================================
# 1. 加载数据
# ============================================================================
print('步骤 1: 加载数据...')
print('-'*80)

data = {}
a_ice_data = {}  # 海冰密集度数据
for exp in ['pi'] + list(EXPERIMENTS.keys()):
    print(f'  加载 {exp.upper()}...')
    file_path = f'{exp}/surface_density_tendency_{exp}_reg.nc'
    data[exp] = xr.open_dataset(file_path)

    # 加载海冰密集度数据
    a_ice_file = f'{exp}/a_ice_reg.nc'
    a_ice_data[exp] = xr.open_dataset(a_ice_file)
    print(f'    已加载海冰数据: {a_ice_file}')

print('✓ 数据加载完成')
print()

# ============================================================================
# 2. 提取并计算JJA平均
# ============================================================================
print('步骤 2: 计算JJA平均...')
print('-'*80)

def get_jja_mean(ds, var_name):
    """提取JJA平均 (6,7,8月)"""
    if var_name not in ds:
        print(f'  警告: {var_name} 不在数据集中')
        return None

    # 提取表层数据 (depth_coord=0)
    var = ds[var_name].isel(depth_coord=0)

    # JJA平均
    jja = var.isel(time=JJA_MONTHS).mean(dim='time')

    return jja

# 为每个试验计算JJA平均
jja_data = {}

for exp in ['pi'] + list(EXPERIMENTS.keys()):
    print(f'  处理 {exp.upper()}...')
    ds = data[exp]

    jja_data[exp] = {}

    # 热通量预算 (读取各分量并乘以-1改变符号)
    jja_data[exp]['heat_sw'] = -get_jja_mean(ds, 'surface_exchange_flux_nonadvective_sw')
    jja_data[exp]['heat_lw'] = -get_jja_mean(ds, 'surface_exchange_flux_nonadvective_lw')
    jja_data[exp]['heat_lh'] = -get_jja_mean(ds, 'surface_exchange_flux_nonadvective_lh')
    jja_data[exp]['heat_sh'] = -get_jja_mean(ds, 'surface_exchange_flux_nonadvective_sh')
    # 总和 = sw + lw + lh + sh
    jja_data[exp]['heat_total'] = (jja_data[exp]['heat_sw'] + jja_data[exp]['heat_lw'] +
                                     jja_data[exp]['heat_lh'] + jja_data[exp]['heat_sh'])

    # 盐度预算 (淡水通量引起的盐度趋势 - 读取各分项并求总和)
    jja_data[exp]['mass_seaice'] = get_jja_mean(ds, 'salt_rhs_sum_surface_ocean_flux_advective_negative_rhs_sum_sea_ice_melt')
    jja_data[exp]['mass_rain'] = get_jja_mean(ds, 'salt_rhs_sum_surface_ocean_flux_advective_negative_rhs_sum_rain_and_ice')
    jja_data[exp]['mass_snow'] = get_jja_mean(ds, 'salt_rhs_sum_surface_ocean_flux_advective_negative_rhs_sum_snow')
    jja_data[exp]['mass_evap'] = get_jja_mean(ds, 'salt_rhs_sum_surface_ocean_flux_advective_negative_rhs_sum_evaporation')
    jja_data[exp]['mass_river'] = get_jja_mean(ds, 'salt_rhs_sum_surface_ocean_flux_advective_negative_rhs_sum_rivers')

    # 加载并计算JJA平均的海冰密集度
    a_ice_jja = a_ice_data[exp]['a_ice'].isel(time=JJA_MONTHS).mean(dim='time')

    # 应用海冰mask到sea ice salt flux: a_ice=0时,mass_seaice=0
    if jja_data[exp]['mass_seaice'] is not None:
        # 创建mask: 只保留a_ice > 0的区域
        ice_mask = a_ice_jja > 0
        # 应用mask (a_ice=0的地方设为0)
        jja_data[exp]['mass_seaice'] = jja_data[exp]['mass_seaice'].where(ice_mask, 0)
        print(f'    已对 mass_seaice 应用海冰mask (a_ice > 0)')

    # 将river变量中的NaN替换为0
    if jja_data[exp]['mass_river'] is not None:
        jja_data[exp]['mass_river'] = jja_data[exp]['mass_river'].fillna(0)

    # 总淡水通量盐度趋势 = 各分项总和
    jja_data[exp]['mass_total'] = (jja_data[exp]['mass_seaice'] +
                                     jja_data[exp]['mass_rain'] +
                                     jja_data[exp]['mass_snow'] +
                                     jja_data[exp]['mass_evap'] +
                                     jja_data[exp]['mass_river'])

    # 组合通量
    jja_data[exp]['heat_radiation'] = jja_data[exp]['heat_sw'] + jja_data[exp]['heat_lw']
    jja_data[exp]['heat_turbulent'] = jja_data[exp]['heat_lh'] + jja_data[exp]['heat_sh']

    # other FW = rain + snow + evap + river
    jja_data[exp]['mass_other'] = (jja_data[exp]['mass_rain'] +
                                     jja_data[exp]['mass_snow'] +
                                     jja_data[exp]['mass_evap'] +
                                     jja_data[exp]['mass_river'])

print('✓ JJA平均计算完成')
print()

# 获取经纬度
lon = data['pi']['lon'].values
lat = data['pi']['lat'].values

# ============================================================================
# 3. 预处理: 平滑数据、添加循环点和创建mesh grid
# ============================================================================
print('步骤 3: 预处理数据平滑、循环点和网格...')
print('-'*80)

def smooth_data(data, sigma=1.5):
    """Apply Gaussian smoothing to data, handling NaN values"""
    # Create a copy and replace NaN with 0 for smoothing
    data_filled = np.nan_to_num(data, nan=0.0)

    # Create a mask for valid data
    mask = ~np.isnan(data)
    mask_float = mask.astype(float)

    # Smooth both data and mask
    smoothed = gaussian_filter(data_filled, sigma=sigma)
    smoothed_mask = gaussian_filter(mask_float, sigma=sigma)

    # Normalize by mask to handle edges properly
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

# 预处理所有数据: 平滑并添加循环点
jja_cyclic = {}
for exp in ['pi'] + list(EXPERIMENTS.keys()):
    jja_cyclic[exp] = {}

    # 计算JJA平均海冰密集度用于mask
    a_ice_jja = a_ice_data[exp]['a_ice'].isel(time=JJA_MONTHS).mean(dim='time')
    # 确保维度正确: 去除单一维度并转为numpy数组
    a_ice_jja = a_ice_jja.squeeze().values  # (180, 360)
    ice_mask = a_ice_jja > 0  # 海冰存在的区域

    for var in ['heat_total', 'heat_radiation', 'heat_turbulent',
                'mass_total', 'mass_seaice', 'mass_other']:
        # 获取数据并确保是2D: (180, 360)
        data_vals = jja_data[exp][var].values
        if data_vals.ndim > 2:
            data_vals = data_vals.squeeze()

        # 对sea ice相关变量应用mask (在平滑之前)
        if var == 'mass_seaice':
            data_vals = np.where(ice_mask, data_vals, 0)

        # 平滑数据
        data_smooth = smooth_data(data_vals, sigma=1.5)
        # 添加循环点
        jja_cyclic[exp][var], lon_cyclic = add_cyclic_point(data_smooth, lon)

# 创建循环mesh grid (只需一次)
lon_2d, lat_2d = np.meshgrid(lon_cyclic, lat)

print('✓ 预处理完成')
print()

# ============================================================================
# 4. 预计算所有异常并确定colorbar范围
# ============================================================================
print('步骤 4: 预计算异常并确定colorbar范围...')
print('-'*80)

# 只统计50°S以南区域
mask_so = lat_2d < -50

anomalies = {}
for exp in EXPERIMENTS.keys():
    anomalies[exp] = {}
    for var in ['heat_total', 'heat_radiation', 'heat_turbulent',
                'mass_total', 'mass_seaice', 'mass_other']:
        anomalies[exp][var] = jja_cyclic[exp][var] - jja_cyclic['pi'][var]

# 根据南大洋区域的实际值范围确定colorbar范围
heat_vars = ['heat_total', 'heat_radiation', 'heat_turbulent']
mass_vars = ['mass_total', 'mass_seaice', 'mass_other']

heat_ranges_anomaly = {}
heat_ranges_pi = {}
mass_ranges_anomaly = {}
mass_ranges_pi = {}

# 异常场范围
for var in heat_vars:
    all_vals = []
    for exp in EXPERIMENTS.keys():
        vals = anomalies[exp][var][mask_so]
        all_vals.extend(vals[~np.isnan(vals)])
    pct_99 = np.percentile(np.abs(all_vals), 99)
    heat_ranges_anomaly[var] = pct_99
    print(f'  {var} anomaly: ±{pct_99:.1f} kJ s⁻¹ m⁻²')

for var in mass_vars:
    all_vals = []
    for exp in EXPERIMENTS.keys():
        vals = anomalies[exp][var][mask_so]
        all_vals.extend(vals[~np.isnan(vals)])
    pct_99 = np.percentile(np.abs(all_vals), 99)
    mass_ranges_anomaly[var] = pct_99
    print(f'  {var} anomaly: ±{pct_99:.2e} kg s⁻¹ m⁻²')

# PI绝对值范围
print('\nPI绝对值范围:')
for var in heat_vars:
    pi_vals = jja_cyclic['pi'][var][mask_so]
    pi_vals = pi_vals[~np.isnan(pi_vals)]
    vmin, vmax = np.percentile(pi_vals, [1, 99])
    heat_ranges_pi[var] = (vmin, vmax)
    print(f'  {var} PI: [{vmin:.1f}, {vmax:.1f}] kJ s⁻¹ m⁻²')

for var in mass_vars:
    pi_vals = jja_cyclic['pi'][var][mask_so]
    pi_vals = pi_vals[~np.isnan(pi_vals)]
    vmin, vmax = np.percentile(pi_vals, [1, 99])
    mass_ranges_pi[var] = (vmin, vmax)
    print(f'  {var} PI: [{vmin:.2e}, {vmax:.2e}] kg s⁻¹ m⁻²')

print('✓ 异常计算完成')
print()

# ============================================================================
# 5. 绘图函数
# ============================================================================

def get_nice_ticks(vmax, n_ticks=5):
    """
    生成对称的等间距刻度,0放在中间
    例如: vmax=456, n_ticks=5 -> [-500, -250, 0, 250, 500]
    确保刻度之间的视觉间距相等
    """
    import math

    # 计算数量级
    magnitude = 10 ** math.floor(math.log10(vmax))

    # 尝试不同的nice numbers
    nice_numbers = [1, 2, 2.5, 5, 10]

    for nice in nice_numbers:
        candidate = nice * magnitude
        if candidate >= vmax:
            # 找到合适的上限
            vmax_nice = candidate
            break
    else:
        vmax_nice = 10 * magnitude

    # 生成对称的等间距刻度
    # n_ticks=5: [-vmax, -vmax/2, 0, vmax/2, vmax]
    # 确保刻度是线性分布的
    ticks = np.linspace(-vmax_nice, vmax_nice, n_ticks)

    return ticks.tolist(), vmax_nice

def plot_field(ax, lon_2d, lat_2d, data, vmin, vmax, cmap, panel_label, center_zero=False):
    """
    绘制场(色阶) - 使用contourf获得平滑效果
    """
    # 色阶 - 使用contourf而不是pcolormesh获得平滑shading
    # 使用线性等间距levels,确保colorbar刻度均匀
    levels = np.linspace(vmin, vmax, 40)  # 40个等级获得平滑渐变

    # 不使用TwoSlopeNorm,直接使用线性normalization
    # 这样colorbar刻度会均匀分布
    cf = ax.contourf(lon_2d, lat_2d, data,
                     levels=levels,
                     cmap=cmap, extend='both',
                     transform=ccrs.PlateCarree())

    # 海岸线
    ax.coastlines(resolution='110m', linewidth=0.5, color='gray')
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)

    # 添加panel label (a), (b), (c)...在左上角
    ax.text(0.02, 0.98, panel_label, transform=ax.transAxes,
            fontsize=12, weight='bold', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # 网格线
    ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.3, linestyle='--')

    return cf

# ============================================================================
# 6. 图1: 热通量预算 (3行×5列: PI + 4个异常)
# ============================================================================
print('步骤 5: 绘制图1 - 热通量预算...')
print('-'*80)

fig = plt.figure(figsize=(18, 12))
gs = GridSpec(3, 5, figure=fig, hspace=0.35, wspace=0.08,
              left=0.05, right=0.95, top=0.96, bottom=0.05)

# 3个变量
var_keys = ['heat_total', 'heat_radiation', 'heat_turbulent']
var_titles = ['Total Heat', 'SW + LW (Radiation)', 'LH + SH (Turbulent)']

# 4个试验
exp_list = list(EXPERIMENTS.keys())
exp_names = list(EXPERIMENTS.values())

# Panel label counter for Figure 1
panel_idx = 0

# 绘制每行
for row, (var_key, var_title) in enumerate(zip(var_keys, var_titles)):
    print(f'  绘制第{row+1}行: {var_title}...')

    # 共享colorbar的mappable列表
    all_cfs = []
    all_axes = []

    # 计算nice范围用于PI和anomaly (在绘图前计算,确保一致)
    vmin, vmax = heat_ranges_pi[var_key]
    vmax_pi = max(abs(vmin), abs(vmax))
    ticks_pi, vmax_pi_nice = get_nice_ticks(vmax_pi, n_ticks=5)

    ticks_anom, vmax_anom_nice = get_nice_ticks(heat_ranges_anomaly[var_key], n_ticks=5)

    # 第1列: PI绝对值
    ax = fig.add_subplot(gs[row, 0], projection=ccrs.SouthPolarStereo())
    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    all_axes.append(ax)

    pi_data = jja_cyclic['pi'][var_key]

    # Panel label
    panel_label = f'({chr(97+panel_idx)})'  # (a), (b), (c)...
    panel_idx += 1

    # 绘制PI - 使用vmax_pi_nice确保与colorbar一致
    cf_pi = plot_field(
        ax, lon_2d, lat_2d, pi_data,
        vmin=-vmax_pi_nice, vmax=vmax_pi_nice,
        cmap='RdBu_r', panel_label=panel_label, center_zero=True
    )
    all_cfs.append(cf_pi)

    # 第2-5列: 4个异常
    for col, (exp, exp_name) in enumerate(zip(exp_list, exp_names), start=1):
        ax = fig.add_subplot(gs[row, col], projection=ccrs.SouthPolarStereo())
        ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
        all_axes.append(ax)

        # 异常数据
        anomaly = anomalies[exp][var_key]

        # Panel label
        panel_label = f'({chr(97+panel_idx)})'
        panel_idx += 1

        # 绘制异常 - 使用vmax_anom_nice确保与colorbar一致
        cf_anom = plot_field(
            ax, lon_2d, lat_2d, anomaly,
            vmin=-vmax_anom_nice, vmax=vmax_anom_nice,
            cmap='RdBu_r', panel_label=panel_label, center_zero=True
        )
        all_cfs.append(cf_anom)

    # 为每行添加两个colorbar: PI列一个, 异常列一个
    # PI colorbar (跨越第1列)
    pos0 = all_axes[0].get_position()
    cbar_ax_pi = fig.add_axes([pos0.x0, pos0.y0 - 0.05, pos0.width, 0.015])
    cbar_pi = plt.colorbar(all_cfs[0], cax=cbar_ax_pi, orientation='horizontal', ticks=ticks_pi)
    cbar_pi.set_label(f'PI: {var_title} (kJ s⁻¹ m⁻²)', fontsize=9)
    cbar_pi.ax.tick_params(labelsize=8)

    # Anomaly colorbar (跨越第2-5列)
    pos1 = all_axes[1].get_position()
    pos4 = all_axes[4].get_position()
    cbar_ax_anom = fig.add_axes([pos1.x0, pos1.y0 - 0.05, pos4.x1 - pos1.x0, 0.015])
    cbar_anom = plt.colorbar(all_cfs[1], cax=cbar_ax_anom, orientation='horizontal', ticks=ticks_anom)
    cbar_anom.set_label(f'Anomaly: {var_title} (kJ s⁻¹ m⁻²)', fontsize=9)
    cbar_anom.ax.tick_params(labelsize=8)

output_file1 = 'figures/surface_heat_budget_anomalies_jja.pdf'
plt.savefig(output_file1, dpi=200, bbox_inches='tight')
print(f'✓ 图1保存: {output_file1}')
print()

plt.close()

# ============================================================================
# 7. 图2: 淡水(盐度)通量预算 (3行×5列: PI + 4个异常)
# ============================================================================
print('步骤 6: 绘制图2 - 淡水通量预算...')
print('-'*80)

fig = plt.figure(figsize=(18, 12))
gs = GridSpec(3, 5, figure=fig, hspace=0.35, wspace=0.08,
              left=0.05, right=0.95, top=0.96, bottom=0.05)

# 3个变量
var_keys = ['mass_total', 'mass_seaice', 'mass_other']
var_titles = ['Total Freshwater', 'Sea Ice Alone', 'Other FW (P+E+R+S)']

# 4个试验
exp_list = list(EXPERIMENTS.keys())
exp_names = list(EXPERIMENTS.values())

# Panel label counter for Figure 2
panel_idx = 0

# 绘制每行
for row, (var_key, var_title) in enumerate(zip(var_keys, var_titles)):
    print(f'  绘制第{row+1}行: {var_title}...')

    # 共享colorbar的mappable列表
    all_cfs = []
    all_axes = []

    # 计算nice范围用于PI和anomaly (在绘图前计算,确保一致)
    vmin, vmax = mass_ranges_pi[var_key]
    vmax_pi = max(abs(vmin), abs(vmax))
    ticks_pi, vmax_pi_nice = get_nice_ticks(vmax_pi, n_ticks=5)

    ticks_anom, vmax_anom_nice = get_nice_ticks(mass_ranges_anomaly[var_key], n_ticks=5)

    # 第1列: PI绝对值
    ax = fig.add_subplot(gs[row, 0], projection=ccrs.SouthPolarStereo())
    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    all_axes.append(ax)

    pi_data = jja_cyclic['pi'][var_key]

    # Panel label
    panel_label = f'({chr(97+panel_idx)})'  # (a), (b), (c)...
    panel_idx += 1

    # 绘制PI - 使用vmax_pi_nice确保与colorbar一致
    cf_pi = plot_field(
        ax, lon_2d, lat_2d, pi_data,
        vmin=-vmax_pi_nice, vmax=vmax_pi_nice,
        cmap='RdBu_r', panel_label=panel_label, center_zero=True
    )
    all_cfs.append(cf_pi)

    # 第2-5列: 4个异常
    for col, (exp, exp_name) in enumerate(zip(exp_list, exp_names), start=1):
        ax = fig.add_subplot(gs[row, col], projection=ccrs.SouthPolarStereo())
        ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
        all_axes.append(ax)

        # 异常数据
        anomaly = anomalies[exp][var_key]

        # Panel label
        panel_label = f'({chr(97+panel_idx)})'
        panel_idx += 1

        # 绘制异常 - 使用vmax_anom_nice确保与colorbar一致
        cf_anom = plot_field(
            ax, lon_2d, lat_2d, anomaly,
            vmin=-vmax_anom_nice, vmax=vmax_anom_nice,
            cmap='RdBu_r', panel_label=panel_label, center_zero=True
        )
        all_cfs.append(cf_anom)

    # 为每行添加两个colorbar: PI列一个, 异常列一个
    # PI colorbar (跨越第1列)
    pos0 = all_axes[0].get_position()
    cbar_ax_pi = fig.add_axes([pos0.x0, pos0.y0 - 0.05, pos0.width, 0.015])
    cbar_pi = plt.colorbar(all_cfs[0], cax=cbar_ax_pi, orientation='horizontal', ticks=ticks_pi)
    cbar_pi.set_label(f'PI: {var_title} (kg s⁻¹ m⁻²)', fontsize=9)
    cbar_pi.ax.tick_params(labelsize=8)

    # Anomaly colorbar (跨越第2-5列)
    pos1 = all_axes[1].get_position()
    pos4 = all_axes[4].get_position()
    cbar_ax_anom = fig.add_axes([pos1.x0, pos1.y0 - 0.05, pos4.x1 - pos1.x0, 0.015])
    cbar_anom = plt.colorbar(all_cfs[1], cax=cbar_ax_anom, orientation='horizontal', ticks=ticks_anom)
    cbar_anom.set_label(f'Anomaly: {var_title} (kg s⁻¹ m⁻²)', fontsize=9)
    cbar_anom.ax.tick_params(labelsize=8)

output_file2 = 'figures/surface_freshwater_budget_anomalies_jja.pdf'
plt.savefig(output_file2, dpi=200, bbox_inches='tight')
print(f'✓ 图2保存: {output_file2}')
print()

plt.close()

# ============================================================================
# 完成
# ============================================================================
print('='*80)
print('绘图完成!')
print('='*80)
print()
print('输出文件:')
print(f'  1. {output_file1}')
print(f'  2. {output_file2}')
print()

# 统计信息
print('异常值统计 (Paleo - PI, 50°S以南区域平均):')
print('-'*80)

for exp, exp_name in EXPERIMENTS.items():
    print(f'\n{exp_name}:')

    # 热通量
    heat_total_mean = anomalies[exp]['heat_total'][mask_so].mean()
    heat_rad_mean = anomalies[exp]['heat_radiation'][mask_so].mean()
    heat_turb_mean = anomalies[exp]['heat_turbulent'][mask_so].mean()

    # 盐通量
    mass_total_mean = anomalies[exp]['mass_total'][mask_so].mean()
    mass_ice_mean = anomalies[exp]['mass_seaice'][mask_so].mean()
    mass_other_mean = anomalies[exp]['mass_other'][mask_so].mean()

    print(f'  Heat Total:      {heat_total_mean:+.2f} kJ s⁻¹ m⁻²')
    print(f'  Heat Radiation:  {heat_rad_mean:+.2f} kJ s⁻¹ m⁻²')
    print(f'  Heat Turbulent:  {heat_turb_mean:+.2f} kJ s⁻¹ m⁻²')
    print(f'  Mass Total:      {mass_total_mean*1e6:+.2f} ×10⁻⁶ kg s⁻¹ m⁻²')
    print(f'  Mass Sea Ice:    {mass_ice_mean*1e6:+.2f} ×10⁻⁶ kg s⁻¹ m⁻²')
    print(f'  Mass Other:      {mass_other_mean*1e6:+.2f} ×10⁻⁶ kg s⁻¹ m⁻²')
print()
