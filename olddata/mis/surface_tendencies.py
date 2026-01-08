#!/usr/bin/env python3
"""
FESOM Surface Tendencies Calculation (通用版本)

支持不同mesh文件结构:
- PI/MH/LIG: nl维度
- LGM/MIS: nz维度

修正：prec, evap, snow 乘以开放水域覆盖率 (1 - A)
"""

import numpy as np
import xarray as xr
import gsw
from pathlib import Path
from scipy.interpolate import griddata
import warnings
import gc
import sys

print("="*80)
print("FESOM Surface Tendencies Calculation - Universal")
print("="*80)

RHO_REF = 1035.0
CP = 3992.0
RHO_WATER = 1000.0

DATA_DIR = Path(".")
OUTPUT_FILE = "fesom_surface_tendencies.nc"
CHUNK_SIZE = 3

# 确定实验名称（从当前目录路径）
current_dir = Path.cwd().name
print(f"实验: {current_dir}")

# 设置mesh文件路径
if current_dir == 'lgm':
    mesh_file = "/home/a/a270064/bb1029/inputs/mesh_glac1d/fesom.mesh.diag.nc"
elif current_dir == 'mis':
    mesh_file = "/home/a/a270064/bb1029/inputs/mesh_glac1d_38k/fesom.mesh.diag.nc"
else:
    mesh_file = "/home/a/a270064/bb1029/inputs/mesh_core2/fesom.mesh.diag.nc"

print(f"Mesh文件: {mesh_file}")

# ============================================================================
# 1. 加载ECHAM数据
# ============================================================================
print("\n[1/8] 加载ECHAM数据...")

echam_file = Path(f'{current_dir}_echam_clim.nc')
echam_ds = xr.open_dataset(echam_file)

echam_lon = echam_ds['lon'].values
echam_lat = echam_ds['lat'].values
echam_time = echam_ds['time'].values

var92 = -echam_ds['var92'].values
var95 = -echam_ds['var95'].values
var111 = -echam_ds['var111'].values
var120 = -echam_ds['var120'].values

print(f'  ECHAM 网格: lon({len(echam_lon)}) × lat({len(echam_lat)})')

# ============================================================================
# 2. 加载FESOM网格 (自适应维度)
# ============================================================================
print("\n[2/8] 加载FESOM网格...")

mesh = xr.open_dataset(mesh_file)

# 检测nod_area的维度
nod_area_dims = mesh['nod_area'].dims
print(f"  nod_area维度: {nod_area_dims}")

if 'nl' in nod_area_dims:
    areacello_1d = mesh['nod_area'].isel(nl=0).values
    print("  使用nl维度")
elif 'nz' in nod_area_dims:
    areacello_1d = mesh['nod_area'].isel(nz=0).values
    print("  使用nz维度")
else:
    # 如果只有一个维度，直接使用
    areacello_1d = mesh['nod_area'].isel({nod_area_dims[0]: 0}).values
    print(f"  使用{nod_area_dims[0]}维度")

fesom_lon_1d = np.degrees(mesh['nodes'].isel(n2=0).values)
fesom_lat_1d = np.degrees(mesh['nodes'].isel(n2=1).values)
nnodes = len(fesom_lon_1d)
nlevels_nod2D = mesh['nlevels_nod2D'].values

print(f"  FESOM节点数: {nnodes}")

ocean_mask = nlevels_nod2D > 0
print(f"  海洋节点: {ocean_mask.sum()}")

mesh.close()

# ============================================================================
# 3. 加载时间坐标
# ============================================================================
print("\n[3/8] 获取时间坐标...")

sst_data = xr.open_dataset(DATA_DIR / "sst_clim.nc")
time_coord = sst_data['time']
ntime = len(time_coord)
print(f"  时间步数: {ntime}")

# ============================================================================
# 4. 加载海冰覆盖率
# ============================================================================
print("\n[4/8] 加载海冰覆盖率...")

aice_file = DATA_DIR / "a_ice_clim.nc"
aice_data = xr.open_dataset(aice_file)
aice = aice_data['a_ice'].values
print(f"  海冰覆盖率范围: [{np.nanmin(aice):.3f}, {np.nanmax(aice):.3f}]")
aice_data.close()

open_water = 1.0 - aice

# ============================================================================
# 5. 插值ECHAM到FESOM
# ============================================================================
print("\n[5/8] 插值ECHAM到FESOM...")

echam_lon_converted = np.where(echam_lon > 180, echam_lon - 360, echam_lon)
echam_lon_2d, echam_lat_2d = np.meshgrid(echam_lon_converted, echam_lat)
points = np.column_stack([echam_lon_2d.flatten(), echam_lat_2d.flatten()])
xi = np.column_stack([fesom_lon_1d, fesom_lat_1d])

lw_fesom = np.zeros((ntime, nnodes), dtype=np.float32)
sw_fesom = np.zeros((ntime, nnodes), dtype=np.float32)
lh_fesom = np.zeros((ntime, nnodes), dtype=np.float32)
sh_fesom = np.zeros((ntime, nnodes), dtype=np.float32)

for t in range(ntime):
    if (t + 1) % 3 == 0 or t == 0:
        print(f'    时间步 {t+1}/{ntime}...')

    lw_fesom[t, :] = griddata(points, var92[t, :, :].flatten(), xi, method='nearest')
    sw_fesom[t, :] = griddata(points, var95[t, :, :].flatten(), xi, method='nearest')
    lh_fesom[t, :] = griddata(points, var111[t, :, :].flatten(), xi, method='nearest')
    sh_fesom[t, :] = griddata(points, var120[t, :, :].flatten(), xi, method='nearest')

print('  ✓ 插值完成')

del var92, var95, var111, var120, points, xi, echam_lon_2d, echam_lat_2d
echam_ds.close()
gc.collect()

# ============================================================================
# 6. 初始化输出
# ============================================================================
print("\n[6/8] 初始化输出...")

ds_out = xr.Dataset()
ds_out['time'] = time_coord
ds_out['nod2'] = np.arange(nnodes)
ds_out['lon'] = (('nod2',), fesom_lon_1d)
ds_out['lat'] = (('nod2',), fesom_lat_1d)
ds_out['nod_area'] = (('nod2',), areacello_1d)
ds_out['ocean_mask'] = (('nod2',), ocean_mask)
ds_out['a_ice'] = (['time', 'nod2'], aice.astype(np.float32))
ds_out['open_water'] = (['time', 'nod2'], open_water.astype(np.float32))

# 热通量
ds_out['lw'] = (['time', 'nod2'], lw_fesom)
ds_out['sw'] = (['time', 'nod2'], sw_fesom)
ds_out['lh'] = (['time', 'nod2'], lh_fesom)
ds_out['sh'] = (['time', 'nod2'], sh_fesom)
ds_out['total_heat'] = (['time', 'nod2'], lw_fesom + sw_fesom + lh_fesom + sh_fesom)

del lw_fesom, sw_fesom, lh_fesom, sh_fesom
gc.collect()

# 预分配其他变量
for var in ['sigma2', 'alpha', 'beta']:
    ds_out[var] = (['time', 'nod2'], np.full((ntime, nnodes), np.nan, dtype=np.float32))

for var in ['prec', 'evap', 'runoff', 'snow', 'seaice', 'total_freshwater']:
    ds_out[var] = (['time', 'nod2'], np.full((ntime, nnodes), np.nan, dtype=np.float32))

for var in ['temp_tendency_lw', 'temp_tendency_sw', 'temp_tendency_lh', 'temp_tendency_sh',
            'temp_tendency_total_heat']:
    ds_out[var] = (['time', 'nod2'], np.full((ntime, nnodes), np.nan, dtype=np.float32))

for var in ['salt_tendency_prec', 'salt_tendency_evap', 'salt_tendency_runoff',
            'salt_tendency_snow', 'salt_tendency_seaice', 'salt_tendency_total_freshwater']:
    ds_out[var] = (['time', 'nod2'], np.full((ntime, nnodes), np.nan, dtype=np.float32))

for var in ['sigma_tendency_lw', 'sigma_tendency_sw', 'sigma_tendency_lh', 'sigma_tendency_sh',
            'sigma_tendency_total_heat',
            'sigma_tendency_prec', 'sigma_tendency_evap', 'sigma_tendency_runoff',
            'sigma_tendency_snow', 'sigma_tendency_seaice', 'sigma_tendency_total_freshwater',
            'sigma_tendency_total']:
    ds_out[var] = (['time', 'nod2'], np.full((ntime, nnodes), np.nan, dtype=np.float32))

# ============================================================================
# 7. 分块处理
# ============================================================================
print(f"\n[7/8] 分块处理数据...")

n_chunks = int(np.ceil(ntime / CHUNK_SIZE))

for chunk_idx in range(n_chunks):
    t_start = chunk_idx * CHUNK_SIZE
    t_end = min((chunk_idx + 1) * CHUNK_SIZE, ntime)
    t_slice = slice(t_start, t_end)

    print(f"\n  批次 {chunk_idx+1}/{n_chunks}: 时间步 {t_start+1}-{t_end}...")

    # 加载SST/SSS
    sst_chunk = sst_data['sst'].isel(time=t_slice).values
    sss_data_file = xr.open_dataset(DATA_DIR / "sss_clim.nc")
    sss_chunk = sss_data_file['sss'].isel(time=t_slice).values
    sss_data_file.close()

    open_water_chunk = open_water[t_slice, :]

    # 加载淡水通量
    prec_data_file = xr.open_dataset(DATA_DIR / "prec_clim.nc")
    prec_raw = prec_data_file['prec'].isel(time=t_slice).values * RHO_WATER
    prec_data_file.close()

    snow_data_file = xr.open_dataset(DATA_DIR / "snow_clim.nc")
    snow_raw = snow_data_file['snow'].isel(time=t_slice).values * RHO_WATER
    snow_data_file.close()

    evap_data_file = xr.open_dataset(DATA_DIR / "evap_clim.nc")
    evap_raw = evap_data_file['evap'].isel(time=t_slice).values * RHO_WATER
    evap_data_file.close()

    runoff_data_file = xr.open_dataset(DATA_DIR / "runoff_clim.nc")
    runoff_chunk = runoff_data_file['runoff'].isel(time=t_slice).values * RHO_WATER
    runoff_data_file.close()

    # 修正：evap 和 snow 需要乘以开放水域覆盖率 (potential fluxes)
    # prec, runoff 是 effective fluxes，不需要修正
    prec_chunk = prec_raw
    snow_chunk = snow_raw * open_water_chunk
    evap_chunk = evap_raw * open_water_chunk

    fw_data_file = xr.open_dataset(DATA_DIR / "fw_clim.nc")
    fw_chunk = - fw_data_file['fw'].isel(time=t_slice).values * RHO_WATER
    fw_data_file.close()

    seaice_chunk = fw_chunk - (prec_chunk + snow_chunk + evap_chunk + runoff_chunk)
    total_fw_chunk = prec_chunk + evap_chunk + runoff_chunk + snow_chunk + seaice_chunk

    ds_out['prec'].values[t_slice, :] = prec_chunk
    ds_out['snow'].values[t_slice, :] = snow_chunk
    ds_out['evap'].values[t_slice, :] = evap_chunk
    ds_out['runoff'].values[t_slice, :] = runoff_chunk
    ds_out['seaice'].values[t_slice, :] = seaice_chunk
    ds_out['total_freshwater'].values[t_slice, :] = total_fw_chunk

    del fw_chunk, prec_raw, snow_raw, evap_raw

    # 计算密度参数
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')

        nt_chunk = t_end - t_start
        lon_broadcast = np.broadcast_to(fesom_lon_1d, (nt_chunk, nnodes))
        lat_broadcast = np.broadcast_to(fesom_lat_1d, (nt_chunk, nnodes))

        SA = gsw.SA_from_SP(sss_chunk, p=0, lon=lon_broadcast, lat=lat_broadcast)
        CT = gsw.CT_from_t(SA, sst_chunk, p=0)

        sigma2_chunk = gsw.sigma2(SA, CT)
        alpha_chunk = gsw.alpha(SA, CT, p=0)
        beta_chunk = gsw.beta(SA, CT, p=0)

        ds_out['sigma2'].values[t_slice, :] = sigma2_chunk
        ds_out['alpha'].values[t_slice, :] = alpha_chunk
        ds_out['beta'].values[t_slice, :] = beta_chunk

    del lon_broadcast, lat_broadcast, SA, CT

    # 热通量趋势
    lw_chunk = ds_out['lw'].values[t_slice, :]
    sw_chunk = ds_out['sw'].values[t_slice, :]
    lh_chunk = ds_out['lh'].values[t_slice, :]
    sh_chunk = ds_out['sh'].values[t_slice, :]

    ds_out['temp_tendency_lw'].values[t_slice, :] = -lw_chunk / (RHO_REF * CP)
    ds_out['temp_tendency_sw'].values[t_slice, :] = -sw_chunk / (RHO_REF * CP)
    ds_out['temp_tendency_lh'].values[t_slice, :] = -lh_chunk / (RHO_REF * CP)
    ds_out['temp_tendency_sh'].values[t_slice, :] = -sh_chunk / (RHO_REF * CP)
    ds_out['temp_tendency_total_heat'].values[t_slice, :] = (
        ds_out['temp_tendency_lw'].values[t_slice, :] +
        ds_out['temp_tendency_sw'].values[t_slice, :] +
        ds_out['temp_tendency_lh'].values[t_slice, :] +
        ds_out['temp_tendency_sh'].values[t_slice, :]
    )

    ds_out['sigma_tendency_lw'].values[t_slice, :] = RHO_REF * alpha_chunk * lw_chunk / CP
    ds_out['sigma_tendency_sw'].values[t_slice, :] = RHO_REF * alpha_chunk * sw_chunk / CP
    ds_out['sigma_tendency_lh'].values[t_slice, :] = RHO_REF * alpha_chunk * lh_chunk / CP
    ds_out['sigma_tendency_sh'].values[t_slice, :] = RHO_REF * alpha_chunk * sh_chunk / CP
    ds_out['sigma_tendency_total_heat'].values[t_slice, :] = (
        ds_out['sigma_tendency_lw'].values[t_slice, :] +
        ds_out['sigma_tendency_sw'].values[t_slice, :] +
        ds_out['sigma_tendency_lh'].values[t_slice, :] +
        ds_out['sigma_tendency_sh'].values[t_slice, :]
    )

    del lw_chunk, sw_chunk, lh_chunk, sh_chunk

    # 淡水通量趋势
    ds_out['salt_tendency_prec'].values[t_slice, :] = -sss_chunk * prec_chunk
    ds_out['salt_tendency_evap'].values[t_slice, :] = -sss_chunk * evap_chunk
    ds_out['salt_tendency_runoff'].values[t_slice, :] = -sss_chunk * runoff_chunk
    ds_out['salt_tendency_snow'].values[t_slice, :] = -sss_chunk * snow_chunk
    ds_out['salt_tendency_seaice'].values[t_slice, :] = -sss_chunk * seaice_chunk
    ds_out['salt_tendency_total_freshwater'].values[t_slice, :] = (
        ds_out['salt_tendency_prec'].values[t_slice, :] +
        ds_out['salt_tendency_evap'].values[t_slice, :] +
        ds_out['salt_tendency_runoff'].values[t_slice, :] +
        ds_out['salt_tendency_snow'].values[t_slice, :] +
        ds_out['salt_tendency_seaice'].values[t_slice, :]
    )

    ds_out['sigma_tendency_prec'].values[t_slice, :] = -RHO_REF * beta_chunk * sss_chunk * prec_chunk / RHO_WATER
    ds_out['sigma_tendency_evap'].values[t_slice, :] = -RHO_REF * beta_chunk * sss_chunk * evap_chunk / RHO_WATER
    ds_out['sigma_tendency_runoff'].values[t_slice, :] = -RHO_REF * beta_chunk * sss_chunk * runoff_chunk / RHO_WATER
    ds_out['sigma_tendency_snow'].values[t_slice, :] = -RHO_REF * beta_chunk * sss_chunk * snow_chunk / RHO_WATER
    ds_out['sigma_tendency_seaice'].values[t_slice, :] = -RHO_REF * beta_chunk * sss_chunk * seaice_chunk / RHO_WATER
    ds_out['sigma_tendency_total_freshwater'].values[t_slice, :] = (
        ds_out['sigma_tendency_prec'].values[t_slice, :] +
        ds_out['sigma_tendency_evap'].values[t_slice, :] +
        ds_out['sigma_tendency_runoff'].values[t_slice, :] +
        ds_out['sigma_tendency_snow'].values[t_slice, :] +
        ds_out['sigma_tendency_seaice'].values[t_slice, :]
    )

    ds_out['sigma_tendency_total'].values[t_slice, :] = (
        ds_out['sigma_tendency_total_heat'].values[t_slice, :] +
        ds_out['sigma_tendency_total_freshwater'].values[t_slice, :]
    )

    del sst_chunk, sss_chunk, prec_chunk, snow_chunk, evap_chunk, runoff_chunk, seaice_chunk
    del total_fw_chunk, sigma2_chunk, alpha_chunk, beta_chunk, open_water_chunk
    gc.collect()

sst_data.close()

# ============================================================================
# 8. 保存
# ============================================================================
print(f"\n[8/8] 保存数据到 {OUTPUT_FILE}...")

encoding = {var: {'zlib': True, 'complevel': 4} for var in ds_out.data_vars}
ds_out.to_netcdf(OUTPUT_FILE, encoding=encoding)

print(f"\n✓ 完成！数据已保存到 {OUTPUT_FILE}")
print("="*80)
