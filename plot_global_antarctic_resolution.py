#!/usr/bin/env python3
"""
High-Quality Global and Antarctic Resolution Map for Triangular Grid
Two-panel figure: Global view + Antarctic focus (50°S-90°S) for AABW research
Publication-ready visualization with professional styling
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.util as cutil
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality parameters
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Load data
print("Loading data...")
ds = xr.open_dataset('reso_reg.nc')
reso = ds['reso'].values[0, 0, :, :]
lat = ds['lat'].values
lon = ds['lon'].values

# Convert resolution to km for better readability
reso_km = reso / 1000.0

# Add cyclic point to avoid discontinuity at 0°/180° longitude
reso_km_cyclic, lon_cyclic = cutil.add_cyclic_point(reso_km, coord=lon)

print(f"Global resolution range: {np.nanmin(reso_km):.1f} - {np.nanmax(reso_km):.1f} km")
print(f"Global mean resolution: {np.nanmean(reso_km):.1f} km")
print(f"Cyclic point added: lon shape {lon.shape} -> {lon_cyclic.shape}")

# Extract Antarctic region (50°S and southward)
antarctic_mask = lat <= -50
antarctic_reso = reso_km[antarctic_mask, :]
antarctic_lat = lat[antarctic_mask]
print(f"\nAntarctic (50°S-90°S) resolution range: {np.nanmin(antarctic_reso):.1f} - {np.nanmax(antarctic_reso):.1f} km")
print(f"Antarctic mean resolution: {np.nanmean(antarctic_reso):.1f} km")

# Create custom colormap (professional blue-green-yellow-red)
colors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0',
          '#fefebb', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#bd0026']
colors.reverse()
n_bins = 256
cmap = LinearSegmentedColormap.from_list('resolution', colors, N=n_bins)

# Use same color scale for both panels
vmin, vmax = np.nanpercentile(reso_km, [2, 98])

# ==================== Create Two-Panel Figure ====================
# Use GridSpec for better control of subplot sizes
fig = plt.figure(figsize=(20, 10), dpi=150)
gs = GridSpec(1, 2, figure=fig, width_ratios=[1.8, 1], wspace=0.15)

# Panel (a): Global view with Robinson projection
ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.Robinson(central_longitude=0))

im1 = ax1.pcolormesh(lon_cyclic, lat, reso_km_cyclic,
                     transform=ccrs.PlateCarree(),
                     cmap=cmap,
                     vmin=vmin,
                     vmax=vmax,
                     shading='auto',
                     rasterized=True)

# Add geographic features
ax1.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', alpha=0.7)
ax1.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray', alpha=0.5)

# Add gridlines
gl1 = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                    linewidth=0.5, color='gray', alpha=0.3, linestyle='--')
gl1.xlines = True
gl1.ylines = True

# Add title with panel label
ax1.set_title('(a) Global Resolution',
              fontsize=16, fontweight='bold', pad=15, loc='left')

# Add statistics text box
#global_stats = f'Min: {np.nanmin(reso_km):.1f} km\nMean: {np.nanmean(reso_km):.1f} km\nMax: {np.nanmax(reso_km):.1f} km'
#ax1.text(0.02, 0.98, global_stats,
#         transform=ax1.transAxes,
#         fontsize=10,
#         verticalalignment='top',
#         bbox=dict(boxstyle='round', facecolor='white', alpha=0.95,
#                  edgecolor='black', linewidth=1.5),
#         fontfamily='monospace')

# Panel (b): Antarctic view with South Polar Stereographic projection (circular)
ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.SouthPolarStereo())

# Set extent to show 50°S and southward
ax2.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())

# Make the polar plot circular by setting boundary
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = plt.matplotlib.path.Path(verts * radius + center)
ax2.set_boundary(circle, transform=ax2.transAxes)

# Use contourf for better handling of cyclic data in polar projection
levels = np.linspace(vmin, vmax, 50)
im2 = ax2.contourf(lon_cyclic, lat, reso_km_cyclic,
                   levels=levels,
                   transform=ccrs.PlateCarree(),
                   cmap=cmap,
                   vmin=vmin,
                   vmax=vmax,
                   extend='both')

# Add geographic features with better visibility for Antarctic
ax2.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='black', alpha=0.8)
ax2.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)

# Add gridlines with latitude circles
#gl2 = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
#                    linewidth=0.8, color='gray', alpha=0.5, linestyle='--')
#gl2.xlines = True
#gl2.ylines = True
#gl2.xlabel_style = {'size': 10, 'color': 'black'}
#gl2.ylabel_style = {'size': 10, 'color': 'black'}

# Add title with panel label
ax2.set_title('(b) Antarctic Region (50°S-90°S)',
              fontsize=16, fontweight='bold', pad=15, loc='left')

# Add Antarctic-specific statistics
#antarctic_stats = f'Min: {np.nanmin(antarctic_reso):.1f} km\nMean: {np.nanmean(antarctic_reso):.1f} km\nMax: {np.nanmax(antarctic_reso):.1f} km'
#ax2.text(0.02, 0.98, antarctic_stats,
#         transform=ax2.transAxes,
#         fontsize=10,
#         verticalalignment='top',
#         bbox=dict(boxstyle='round', facecolor='white', alpha=0.95,
#                  edgecolor='black', linewidth=1.5),
#         fontfamily='monospace')

# Add AABW label
#ax2.text(0.98, 0.98, 'AABW\nResearch\nRegion',
#         transform=ax2.transAxes,
#         fontsize=11,
#         fontweight='bold',
#         verticalalignment='top',
#         horizontalalignment='right',
#         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8,
#                  edgecolor='darkblue', linewidth=2),
#         color='darkblue')

# Add shared colorbar at the bottom
cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.03])
cbar = plt.colorbar(im1, cax=cbar_ax, orientation='horizontal', extend='both')
cbar.set_label('Grid Resolution (km)', fontsize=14, fontweight='bold', labelpad=10)
cbar.ax.tick_params(labelsize=11, width=1.5, length=6)

# Add overall title
#fig.suptitle('Grid Resolution: Global and Antarctic Focus',
#             fontsize=20, fontweight='bold', y=0.98)

# Adjust layout (removed since we're using GridSpec)
# plt.subplots_adjust(top=0.93, bottom=0.15, left=0.05, right=0.95, wspace=0.15)

# Save outputs
output_png = 'global_antarctic_resolution.png'
plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"\n✓ High-quality PNG saved to: {output_png}")

output_pdf = 'figures/global_antarctic_resolution.pdf'
plt.savefig(output_pdf, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"✓ Vector PDF saved to: {output_pdf}")

print("\n" + "="*60)
print("Successfully generated two-panel visualization:")
print(f"  • Left panel: Global resolution (Robinson projection)")
print(f"  • Right panel: Antarctic focus 50°S-90°S (Polar Stereographic)")
print(f"\nFiles created:")
print(f"  1. {output_png}")
print(f"  2. {output_pdf}")
print("="*60)
