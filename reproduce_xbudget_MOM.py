#!/usr/bin/env python
"""
复现 xbudget 示例: MOM6 质量、热量和盐度预算

此脚本复现 xbudget/examples/MOM6_budget_examples_mass_heat_salt.ipynb 的内容
分析有限体积海洋模式的质量和示踪剂预算闭合
"""

import xarray as xr
import xgcm
import matplotlib.pyplot as plt
import json
import numpy as np
import sys
import os

# 导入 xbudget
import xbudget

print('='*80)
print('xbudget MOM6 预算分析')
print('='*80)
print()

# ============================================================================
# 辅助函数: 构建网格
# ============================================================================
def construct_grid(ds):
    """构建 xgcm Grid 对象"""
    coords = {
        'X': {'center': 'xh', 'outer': 'xq'},
        'Y': {'center': 'yh', 'outer': 'yq'},
        'Z': {'center': 'z_l', 'outer': 'z_i'},
    }
    boundary = {'X': 'periodic', 'Y': 'extend', 'Z': 'extend'}
    metrics = {('X', 'Y'): 'areacello'}
    grid = xgcm.Grid(ds, coords=coords, metrics=metrics, boundary=boundary, autoparse_metadata=False)
    return grid

# ============================================================================
# 1. 加载数据
# ============================================================================
print('步骤 1: 加载 MOM6 3D 诊断数据 (z-level 坐标)...')
print('-'*80)

data_file = '/work/ba0989/a270064/bb1029/wmt/xbudget/data/MOM6_global_example_diagnostics_zlevels_v0_0_6.nc'
print(f'数据文件: {data_file}')
print(f'文件大小: {os.path.getsize(data_file) / 1024 / 1024:.1f} MB')
print()

# 加载数据
ds = xr.open_dataset(data_file, chunks=-1).fillna(0.)
grid = construct_grid(ds)

print(f'数据集维度:')
print(f'  - 水平: xh={ds.dims["xh"]}, yh={ds.dims["yh"]}')
print(f'  - 垂直: z_l={ds.dims["z_l"]} 层 (z_i={ds.dims["z_i"]} 界面)')
print(f'  - 时间: {ds.dims["time"]} 时间步')
print(f'数据集变量数量: {len(ds.data_vars)}')
print()

# ============================================================================
# 2. 加载预算元数据
# ============================================================================
print('步骤 2: 从预设 MOM6 字典加载预算元数据...')
print('-'*80)

xbudget_dict = xbudget.load_preset_budget(model="MOM6").copy()

print('预算类型:')
for budget_name in xbudget_dict.keys():
    print(f'  - {budget_name}')
print()

# ============================================================================
# 3. 收集预算项
# ============================================================================
print('步骤 3: 使用 xbudget 收集和重构预算项...')
print('-'*80)

xbudget.collect_budgets(grid, xbudget_dict)

print('✓ 预算项收集完成')
print()

# 打印热量预算结构示例
print('热量预算结构示例 (部分):')
print('LHS (左侧趋势项):')
from xbudget import get_vars
lhs_vars = get_vars(xbudget_dict, 'heat_lhs')
if 'sum' in lhs_vars:
    for term in lhs_vars['sum'][:3]:
        print(f'  - {term}')
    print(f'  ... 等')

print('\nRHS (右侧源汇项):')
rhs_vars = get_vars(xbudget_dict, 'heat_rhs')
if 'sum' in rhs_vars:
    for term in rhs_vars['sum'][:5]:
        print(f'  - {term}')
    print(f'  ... 等')
print()

# ============================================================================
# 4. 验证预算闭合
# ============================================================================
print('步骤 4: 验证预算闭合 (LHS vs RHS)...')
print('-'*80)

fig, axes = plt.subplots(3, 2, figsize=(12, 14))

for idx, (eq, vmax) in enumerate(zip(["mass", "heat", "salt"], [1.e4, 1e12, 1.e5])):
    # LHS
    ax = axes[idx, 0]
    lhs_var = get_vars(xbudget_dict, f"{eq}_lhs")['var']
    data_lhs = grid._ds[lhs_var].isel(z_l=0).isel(time=0)
    im = data_lhs.plot(ax=ax, vmin=-vmax, vmax=vmax, cmap="RdBu_r", add_colorbar=True)
    ax.set_title(f"LHS {eq} tendency (surface layer)")

    # RHS
    ax = axes[idx, 1]
    rhs_var = get_vars(xbudget_dict, f"{eq}_rhs")['var']
    data_rhs = grid._ds[rhs_var].isel(z_l=0).isel(time=0)
    im = data_rhs.plot(ax=ax, vmin=-vmax, vmax=vmax, cmap="RdBu_r", add_colorbar=True)
    ax.set_title(f"RHS {eq} tendency (surface layer)")

plt.tight_layout()
output_budget_closure = 'xbudget_budget_closure.png'
plt.savefig(output_budget_closure, dpi=150, bbox_inches='tight')
print(f'✓ 保存预算闭合验证图: {output_budget_closure}')
print()

# ============================================================================
# 5. 简化预算聚合
# ============================================================================
print('步骤 5: 生成简化的高级预算聚合...')
print('-'*80)

simple_budgets = xbudget.aggregate(xbudget_dict)

print('简化预算结构:')
for budget_name, budget_dict in simple_budgets.items():
    print(f'\n{budget_name.upper()} 预算:')
    print(f'  Lambda: {budget_dict.get("lambda", "N/A")}')
    if 'surface_lambda' in budget_dict:
        print(f'  Surface Lambda: {budget_dict["surface_lambda"]}')
    print(f'  LHS 项数: {len(budget_dict.get("lhs", {}))}')
    print(f'  RHS 项数: {len(budget_dict.get("rhs", {}))}')
    print(f'  RHS 项:')
    for term_name in budget_dict.get("rhs", {}).keys():
        print(f'    - {term_name}')
print()

# 绘制简化预算的 RHS 项
fig = plt.figure(figsize=(16, 12))

for budget_idx, (eq, vmax) in enumerate(zip(["mass", "heat", "salt"], [1e6, 1e12, 1e4])):
    N = len(simple_budgets[eq]['rhs'])

    for i, (k, v) in enumerate(simple_budgets[eq]['rhs'].items(), start=1):
        ax = plt.subplot(3, max(3, N), budget_idx * max(3, N) + i)

        if "z_l" in grid._ds[v].dims:
            grid._ds[v].isel(z_l=0).isel(time=0).plot(ax=ax, vmin=-vmax, vmax=vmax,
                                                       cmap="RdBu_r", add_colorbar=True)
        else:
            grid._ds[v].isel(time=0).plot(ax=ax, vmin=-vmax, vmax=vmax,
                                         cmap="RdBu_r", add_colorbar=True)
        ax.set_title(f"{eq} {k}", fontsize=9)

plt.tight_layout()
output_simple_budgets = 'xbudget_simple_budgets.png'
plt.savefig(output_simple_budgets, dpi=150, bbox_inches='tight')
print(f'✓ 保存简化预算图: {output_simple_budgets}')
print()

# ============================================================================
# 6. 自定义预算分解
# ============================================================================
print('步骤 6: 生成自定义预算分解...')
print('-'*80)

decompose_list = ["surface_exchange_flux", "nonadvective", "diffusion"]
print(f'分解项: {", ".join(decompose_list)}')
print()

decomposed_budgets = xbudget.aggregate(xbudget_dict, decompose=decompose_list)

print('分解后的预算结构:')
for budget_name, budget_dict in decomposed_budgets.items():
    print(f'\n{budget_name.upper()} 预算:')
    print(f'  RHS 项数: {len(budget_dict.get("rhs", {}))}')
    print(f'  RHS 分解项:')
    for term_name in list(budget_dict.get("rhs", {}).keys())[:5]:
        print(f'    - {term_name}')
    if len(budget_dict.get("rhs", {})) > 5:
        print(f'    ... 还有 {len(budget_dict.get("rhs", {})) - 5} 项')
print()

# 绘制分解后的预算
fig = plt.figure(figsize=(16, 18))

for budget_idx, (eq, vmax) in enumerate(zip(["mass", "heat", "salt"], [1e6, 1e12, 1e4])):
    nterms = len(decomposed_budgets[eq]['rhs'])
    ncols = min(4, nterms)
    nrows = int(np.ceil(nterms / ncols))

    for i, (k, v) in enumerate(decomposed_budgets[eq]['rhs'].items(), start=1):
        ax = plt.subplot(3 * nrows, ncols, budget_idx * nrows * ncols + i)

        if "z_l" in grid._ds[v].dims:
            grid._ds[v].isel(z_l=0).isel(time=0).plot(ax=ax, vmin=-vmax, vmax=vmax,
                                                       cmap="RdBu_r", add_colorbar=False)
        else:
            grid._ds[v].isel(time=0).plot(ax=ax, vmin=-vmax, vmax=vmax,
                                         cmap="RdBu_r", add_colorbar=False)
        ax.set_title(f"{eq} {k}", fontsize=7)

plt.tight_layout()
output_decomposed = 'xbudget_decomposed_budgets.png'
plt.savefig(output_decomposed, dpi=150, bbox_inches='tight')
print(f'✓ 保存分解预算图: {output_decomposed}')
print()

# ============================================================================
# 7. 保存分解后的预算到 NetCDF
# ============================================================================
print('步骤 7: 保存分解后的预算数据...')
print('-'*80)

# 收集所有要保存的变量名
vars_to_save = []

# 添加 lambda 变量
for budget_name, budget_dict in decomposed_budgets.items():
    if 'lambda' in budget_dict:
        vars_to_save.append(budget_dict['lambda'])
    if 'surface_lambda' in budget_dict:
        vars_to_save.append(budget_dict['surface_lambda'])

# 添加所有预算项
for budget_name, budget_dict in decomposed_budgets.items():
    for side in ['lhs', 'rhs']:
        if side in budget_dict:
            for term_name, var_name in budget_dict[side].items():
                vars_to_save.append(var_name)

# 去重
vars_to_save = list(set(vars_to_save))

# 过滤掉不存在的变量
vars_to_save = [v for v in vars_to_save if v in grid._ds]

# 从grid._ds中选择这些变量
output_ds = grid._ds[vars_to_save]

output_nc = 'xbudget_mass_heat_salt_budgets.nc'
output_ds.to_netcdf(output_nc)
print(f'✓ 保存到 {output_nc}')
print(f'文件大小: {os.path.getsize(output_nc) / 1024 / 1024:.1f} MB')
print()

# ============================================================================
# 8. 生成变量使用汇总
# ============================================================================
print('='*80)
print('数据变量使用分析总结')
print('='*80)
print()

print(f'使用的数据文件:')
print(f'  {data_file}')
print()

print('关键输入变量分类:')
print()

# 收集所有使用的原始变量
used_vars = set()

def extract_vars_from_dict(d):
    """递归提取字典中的所有变量名"""
    if isinstance(d, dict):
        for k, v in d.items():
            if k in ['tracer_content_tendency_per_unit_area', 'thickness_tendency',
                     'mass_tendency_per_unit_area', 'lambda_mass', 'area']:
                if isinstance(v, str) and v in ds:
                    used_vars.add(v)
            if isinstance(v, dict):
                extract_vars_from_dict(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        extract_vars_from_dict(item)

extract_vars_from_dict(xbudget.load_preset_budget(model="MOM6"))

# 分类显示
print('1. 示踪剂场 (Tracer Fields):')
for var in sorted(['thetao', 'so', 'tos', 'sos']):
    if var in ds:
        v = ds[var]
        units = v.attrs.get('units', 'N/A')
        dims = str(v.dims)
        print(f'   {var:30s} | {units:15s} | {dims}')
print()

print('2. 网格和几何信息:')
for var in sorted(['areacello', 'volcello', 'thkcello', 'wet', 'deptho']):
    if var in ds:
        v = ds[var]
        units = v.attrs.get('units', 'N/A')
        dims = str(v.dims)
        print(f'   {var:30s} | {units:15s} | {dims}')
print()

print('3. 趋势诊断 (Tendency Diagnostics):')
tendency_vars = [v for v in used_vars if 'tend' in v or 'dhdt' in v]
for var in sorted(tendency_vars)[:10]:
    if var in ds:
        v = ds[var]
        units = v.attrs.get('units', 'N/A')
        dims_str = f"({', '.join(v.dims)})"
        print(f'   {var:30s} | {units:15s} | {dims_str}')
if len(tendency_vars) > 10:
    print(f'   ... 还有 {len(tendency_vars) - 10} 个趋势变量')
print()

print('4. 平流和扩散诊断:')
for var in sorted(['T_advection_xy', 'Th_tendency_vert_remap', 'S_advection_xy',
                   'Sh_tendency_vert_remap', 'opottempdiff', 'opottemppmdiff',
                   'osaltdiff', 'osaltpmdiff']):
    if var in ds:
        v = ds[var]
        units = v.attrs.get('units', 'N/A')
        print(f'   {var:30s} | {units:15s}')
print()

print('5. 表面通量:')
for var in sorted(['hflso', 'hfsso', 'rlntds', 'rsdoabsorb', 'heat_content_surfwater',
                   'sfdsi', 'evs', 'prlq', 'prsn', 'friver', 'ficeberg', 'fsitherm', 'vprec', 'wfo']):
    if var in ds:
        v = ds[var]
        units = v.attrs.get('units', 'N/A')
        print(f'   {var:30s} | {units:15s}')
print()

print('='*80)
print('脚本执行完成!')
print('='*80)
print()
print('生成的文件:')
print(f'  - {output_budget_closure}')
print(f'  - {output_simple_budgets}')
print(f'  - {output_decomposed}')
print(f'  - {output_nc}')
