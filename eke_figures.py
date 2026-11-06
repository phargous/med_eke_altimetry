"""
Plotting functions for the figures of Hargous et al. (2026), Ocean Science 22, 549-564.

One function per figure; each returns the matplotlib ``Figure``. Computations
are done in ``eke_analysis.py``.
"""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from eke_analysis import ALPHA, REGION_NAMES

# --------------------------------------------------------------------------
# Common style
# --------------------------------------------------------------------------
PRODUCTS = {  # key: (label, line colour, bar face colour, bar edge colour)
    'AM': ('all-sat-euro', 'red', 'lightcoral', 'indianred'),
    'AG': ('all-sat-glo', 'orange', 'gold', 'goldenrod'),
    'TG': ('two-sat-glo', 'blue', 'cornflowerblue', 'royalblue'),
}


def land(scale='50m', linewidth=0.5):
    return cfeature.NaturalEarthFeature('physical', 'land', scale, edgecolor='black',
                                        facecolor='navajowhite', linewidth=linewidth)


def _gridlines(ax, size=20, left=True, bottom=True):
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle='--', color='gray')
    gl.top_labels = gl.right_labels = False
    gl.left_labels, gl.bottom_labels = left, bottom
    gl.xlabel_style = gl.ylabel_style = {'size': size}
    return gl


def _bold_title(ax, letter, text='', size=15, loc='left'):
    ax.set_title(rf'$\bf{{({letter})}}$ {text}'.rstrip(), fontsize=size, loc=loc)


def _year_axis(ax, major=4):
    ax.xaxis.set_major_locator(mdates.YearLocator(major))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.YearLocator())
    ax.grid(True, which='major', linewidth=1)
    ax.grid(True, which='minor', linewidth=0.5)


# --------------------------------------------------------------------------
# Figure 1: mean EKE map
# --------------------------------------------------------------------------
def fig1_mean_eke(mean_eke, boxes=True, vmax=500):
    """Temporal mean EKE of all-sat-euro (1993-2023)."""
    fig, ax = plt.subplots(figsize=(25, 10), subplot_kw={'projection': ccrs.Mercator()},
                           constrained_layout=True)
    ax.set_extent([-10, 37, 30, 47], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {'size': 20, 'color': 'black'}
    cs = ax.contourf(mean_eke.lon.values, mean_eke.lat.values, mean_eke.values,
                     levels=np.linspace(0, vmax, 100), vmin=0, vmax=vmax, cmap='inferno',
                     transform=ccrs.PlateCarree())
    iso = ax.contour(mean_eke.lon.values, mean_eke.lat.values, mean_eke.values,
                     levels=[100, 200, 300, 500], colors='white', linewidths=0.7,
                     transform=ccrs.PlateCarree())
    ax.clabel(iso, inline=True, fontsize=12, fmt='%.0f')
    ax.add_feature(land())
    if boxes:
        for (x0, y0, w, h), color, name in [((-5.4, 35, 4.6, 2), 'red', 'Alboran Sea'),
                                            ((25, 31, 5, 5), 'blue', 'Ierapetra area')]:
            ax.add_patch(Rectangle((x0, y0), w, h, linewidth=2, edgecolor=color, facecolor='none',
                                   transform=ccrs.PlateCarree(), zorder=100))
            ax.text(x0, y0 - 0.5, name, color=color, fontsize=20, transform=ccrs.PlateCarree(), zorder=100)
    cbar = fig.colorbar(cs, ax=ax, orientation='vertical', fraction=0.015, pad=0.04)
    cbar.set_label('EKE (cm$^2$ s$^{-2}$)', fontsize=25)
    cbar.set_ticks(np.linspace(0, vmax, 6))
    cbar.ax.tick_params(labelsize=25)
    return fig


# --------------------------------------------------------------------------
# Figure 2: ground tracks and number of satellites
# --------------------------------------------------------------------------
def _track_panel(ax, tracks, extent, labels_left=True):
    ax.coastlines()
    ax.add_feature(cfeature.LAND, facecolor='navajowhite')
    gl = _gridlines(ax, left=labels_left)
    gl.ylocator = mticker.MultipleLocator(0.5)
    for lon, lat in tracks:
        ax.scatter(lon, lat, s=20, c='black', transform=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())


def fig2_tracks_and_satellites(tracks_two, tracks_all, n_sat, extent=(-6, -1, 35, 37)):
    """(a) two-sat tracks, (b) all-sat tracks over 20 days, (c) number of altimeters.

    ``tracks_two`` / ``tracks_all`` are lists of ``(lon, lat)`` arrays, ``n_sat`` a
    daily pandas Series.
    """
    fig = plt.figure(figsize=(25, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.Mercator())
    ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.Mercator())
    ax3 = fig.add_subplot(gs[1, :])
    _track_panel(ax1, tracks_two, extent)
    ax1.set_title('(a)', fontsize=20, fontweight='bold')
    _track_panel(ax2, tracks_all, extent, labels_left=False)
    ax2.set_title('(b)', fontsize=20, fontweight='bold')

    ax3.plot(n_sat.index, n_sat.values, color='black', lw=4, label='number of satellites')
    _year_axis(ax3)
    ax3.set_xlabel('Time', fontsize=20)
    ax3.set_ylabel('Number of satellites', fontsize=20)
    ax3.tick_params(axis='both', labelsize=20)
    ax3.set_title('(c)', fontweight='bold', fontsize=20)
    ax3.legend(fontsize=20)
    return fig


# --------------------------------------------------------------------------
# Figure 3: regional time series and trends
# --------------------------------------------------------------------------
INSET_SETTINGS = {  # extent, inset position [left, bottom, width, height] in axes fractions
    'MED': ([-10, 40, 28, 47], [0.518, 1, 0.5, 0.5]),
    'ALB': ([-8, 10, 35, 42], [0.587, 1, 0.5, 0.5]),
    'ALBB': ([-8, 10, 35, 42], [0.518, 0.7, 0.5, 0.5]),
    'IERA': ([20, 37, 31, 38], [0.587, 0.7, 0.5, 0.5]),
}


def _add_inset(fig, ax, grid, mask, extent, position, track_lon, track_lat):
    box = ax.get_position()
    x0 = box.x0 + position[0] * box.width
    y0 = box.y0 + position[1] * box.height
    inset = fig.add_axes([x0, y0, position[2] * box.width, position[3] * box.height],
                         projection=ccrs.Mercator())
    inset.set_extent(extent, crs=ccrs.PlateCarree())
    # drawing order: mask (1) < land (2) < coastline (3) < tracks (4)
    inset.contourf(grid.lon.values, grid.lat.values, np.where(mask, 1.0, np.nan),
                   levels=[0.5, 1.5], colors=['magenta'], transform=ccrs.PlateCarree(), zorder=1)
    inset.add_feature(cfeature.LAND, color='navajowhite', zorder=2)
    inset.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    inset.scatter(track_lon, track_lat, s=0.5, c='black', transform=ccrs.PlateCarree(), zorder=4)


def fig3_regional_series(series, trends, masks, grids, ref_tracks,
                         regions=('MED', 'ALB', 'ALBB', 'IERA'), inset_settings=INSET_SETTINGS):
    """Area-weighted mean EKE of the three L4 products over four regions, and their trends.

    Parameters
    ----------
    series : ``{region: {product: DataArray}}``
    trends : ``{region: {product: eke_analysis.Trend}}``
    masks : output of ``load_region_masks`` (used for the insets)
    grids : ``{'med': dataset, 'glo': dataset}`` providing lat/lon
    ref_tracks : (lon, lat) arrays of reference altimeter tracks for the insets
    """
    fig = plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(3, 2, figure=fig)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    ax_bar = fig.add_subplot(gs[2, :])
    letters = 'abcd'

    for k, (ax, reg) in enumerate(zip(axes, regions)):
        for p, (label, color, _, _) in PRODUCTS.items():
            s = series[reg][p]
            ax.plot(s.time, s, color=color, lw=1, alpha=0.2)
            ax.plot(s.time, s.rolling(time=365, center=True).mean(), color=color, lw=1)
            ax.plot(s.time, trends[reg][p].line(), color=color, lw=2, label=label if k == 0 else None)
        ax.set_ylabel('EKE (cm$^2$s$^{-2}$)', fontsize=15)
        _bold_title(ax, letters[k], REGION_NAMES[reg])
        ax.tick_params(axis='both', labelsize=13)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
        if k % 2 == 1:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position('right')
    axes[0].legend(bbox_to_anchor=(1, 1.25), fontsize=15, ncols=3)
    plt.tight_layout()  # fix the layout before placing the insets

    for ax, reg in zip(axes, regions):
        extent, position = inset_settings[reg]
        for_grid = 'med'  # insets show the region on the 1/8 deg grid
        _add_inset(fig, ax, grids[for_grid], masks[reg][for_grid], extent, position, *ref_tracks)

    # (e) bar chart of the trends
    xbar = np.arange(len(regions))
    offset, width = 0.15, 0.25
    ax_bar.set_xticks(xbar)
    ax_bar.set_xticklabels([REGION_NAMES[r] for r in regions], size=15)
    for i, reg in enumerate(regions):
        for shift, (p, (_, color, face, edge)) in zip((-2 * offset, 0, 2 * offset), PRODUCTS.items()):
            t = trends[reg][p]
            ns = not t.significant
            ax_bar.bar(xbar[i] + shift, t.slope, width=width, color=face, edgecolor=edge,
                       hatch='//////' if ns else None)
            ax_bar.errorbar(xbar[i] + shift, t.slope, yerr=t.std, fmt='none', ecolor='black',
                            elinewidth=1, capsize=15)
            ax_bar.text(xbar[i] + shift, -2, f'{t.slope:.02f}\n±{t.std:.02f}', fontsize=15,
                        ha='center', va='bottom', color=color,
                        weight='normal' if ns else 'bold', style='italic' if ns else 'normal')
    ax_bar.set_ylabel('EKE trends (cm$^2$s$^{-2}$year$^{-1}$)', fontsize=15)
    ax_bar.set_title(r'$\bf{(e)}$', fontsize=15)
    return fig


# --------------------------------------------------------------------------
# Figure 4: satellite-count correction
# --------------------------------------------------------------------------
def fig4_satellite_correction(time, n_sat, diff, two_sat, corrected, trend_two, trend_corrected):
    """(a) all-sat minus two-sat EKE vs number of satellites; (b) two-sat vs corrected all-sat."""
    fig, axs = plt.subplots(1, 2, figsize=(25, 7))

    axs[0].plot(n_sat.index, n_sat.values, color='black', zorder=100, lw=4, label='number of satellites')
    ax2 = axs[0].twinx()
    ax2.set_zorder(0)
    ax2.plot(time, diff, color='grey', label='all-sat-glo - two-sat-glo')
    axs[0].set_ylabel('Number of satellites', fontsize=20)
    ax2.set_ylabel('EKE (cm$^2$s$^{-2}$)', fontsize=20)
    axs[0].legend(fontsize=20)
    ax2.legend(fontsize=20, bbox_to_anchor=(0.402, 0.9))
    axs[0].set_title('(a)', fontweight='bold', fontsize=20)
    ax2.tick_params(axis='both', labelsize=15)

    for ts, tr, color, label in [(two_sat, trend_two, 'blue', 'two-sat-glo'),
                                 (corrected, trend_corrected, 'green', 'all-sat-glo-corrected')]:
        axs[1].plot(time, ts, color=color, lw=1, alpha=0.5)
        axs[1].plot(time, ts.rolling(time=365, center=True).mean(), color=color, lw=1)
        axs[1].plot(time, tr.line(), color=color, lw=2,
                    label=f'{label}, trend={tr.slope:.2f} cm$^2$s$^{{-2}}$year$^{{-1}}$')
    axs[1].set_ylabel('EKE (cm$^2$s$^{-2}$)', fontsize=20)
    axs[1].legend(loc='upper left', fontsize=20)
    axs[1].set_title('(b)', fontweight='bold', fontsize=20)

    for ax in axs:
        ax.set_xlabel('Time', fontsize=20)
        ax.tick_params(axis='both', labelsize=15)
        _year_axis(ax)
    plt.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Figure 5: spatial correlation map
# --------------------------------------------------------------------------
def fig5_correlation_map(lon, lat, corr, extent=(-6, 37, 30, 47)):
    """Correlation between all-sat minus two-sat EKE and the number of satellites."""
    fig = plt.figure(figsize=(25, 20))
    ax = plt.subplot(111, projection=ccrs.Mercator())
    ax.add_feature(land())
    _gridlines(ax)
    cs = ax.contourf(lon, lat, corr, levels=10, cmap='coolwarm', vmin=-1, vmax=1,
                     transform=ccrs.PlateCarree())
    ax.set_facecolor('black')  # black = non-significant correlation or land
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    cbar = fig.colorbar(cs, ax=ax, orientation='vertical', fraction=0.015, pad=0.04)
    cbar.set_label('Pearson correlation coefficient', fontsize=20)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_ticks(np.linspace(-1, 1, 11))
    return fig


# --------------------------------------------------------------------------
# Figure 6: trend maps
# --------------------------------------------------------------------------
def fig6_trend_maps(trend_maps, grids, extent=(-6, 36.5, 30, 46), vmin=-9, vmax=8):
    """Gridded EKE trends; hatching marks non-significant trends.

    ``trend_maps`` is ``{'AM': ..., 'AG': ..., 'TG': ...}`` (outputs of
    ``eke_analysis.trend_map``) and ``grids`` maps ``AM`` to the 1/8 deg grid
    and ``AG``/``TG`` to the 1/4 deg grid.
    """
    fig = plt.figure(figsize=(25, 18))
    gs = gridspec.GridSpec(2, 4, hspace=0, wspace=0.05)
    axs = [fig.add_subplot(gs[0, 0:2], projection=ccrs.Mercator()),
           fig.add_subplot(gs[0, 2:], projection=ccrs.Mercator()),
           fig.add_subplot(gs[1, 1:3], projection=ccrs.Mercator())]
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    levels = np.linspace(vmin, vmax, 170)
    for i, (ax, p, letter) in enumerate(zip(axs, ('AM', 'AG', 'TG'), 'abc')):
        g, m = grids[p], trend_maps[p]
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = gl.ylabel_style = {'size': 20, 'color': 'black'}
        if i == 1:
            gl.left_labels = False
        cs = ax.contourf(g.lon.values, g.lat.values, m['slope'], levels=levels, cmap='seismic',
                         norm=norm, transform=ccrs.PlateCarree())
        ax.contourf(g.lon.values, g.lat.values, np.where(m['pvalue'] > ALPHA, 1, np.nan),
                    levels=[0.99, 1.01], colors='none', hatches=['///////'], transform=ccrs.PlateCarree())
        ax.set_facecolor('black')
        ax.add_feature(land())
        ax.set_title(rf'$\bf{{({letter})}}$ {PRODUCTS[p][0]}', fontsize=20)
    cax = fig.add_axes([0.25, 0.5, 0.5, 0.02])
    cbar = fig.colorbar(cs, cax=cax, orientation='horizontal')
    cbar.set_label('EKE trends (cm$^2$ s$^{-2}$ year$^{-1}$)', fontsize=20)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_ticks(np.arange(vmin, vmax + 1, 1))
    return fig


# --------------------------------------------------------------------------
# Figure 7: along-track L3 versus L4 trends
# --------------------------------------------------------------------------
def _trend_profile(ax, tr, title, ylim=(-5, 3.2)):
    """Trend along one track: solid = significant, dashed = all."""
    n = len(tr['lon'])
    x = np.arange(1, n + 1)
    for key, color, label in [('L3', 'green', 'L3-ref'), ('AG', 'orange', 'all-sat-glo')]:
        s, p = tr[key]['slope'], tr[key]['p']
        ax.plot(x, np.where(p < ALPHA, s, np.nan), color=color, lw=2, label=label)
        ax.plot(x, s, color=color, lw=1, ls='--')
    ax.axhline(0, color='black')
    ax.set_ylabel('Trends (cm$^2$s$^{-2}$year$^{-1}$)', fontsize=20)
    ax.tick_params(axis='both', labelsize=20)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0f}' if v % 2 == 1 else ''))
    ax.set_ylim(*ylim)
    ax.grid()
    ax.text(1, ylim[0] * 0.8, title, fontsize=20, color='black')


def _track_scatter(ax, tr, norm):
    return ax.scatter(tr['lon'], tr['lat'], s=1, c=tr['AG']['slope'], norm=norm, cmap='coolwarm',
                      edgecolor=np.where(tr['AG']['p'] < ALPHA, 'none', 'black'), zorder=1,
                      transform=ccrs.PlateCarree())


def _trend_background(ax, grid, trend, norm):
    ax.contourf(grid.lon.values, grid.lat.values, trend['slope'], levels=100, cmap='coolwarm', norm=norm,
                transform=ccrs.PlateCarree())
    ax.contourf(grid.lon.values, grid.lat.values, np.where(trend['pvalue'] > ALPHA, 1, np.nan),
                levels=[0.99, 1.01], colors='none', hatches=['//'], transform=ccrs.PlateCarree())


def _annotate_track(ax, tr, label, k_text, k_arrow, dx_text=0.1, dx_arrow=0.2):
    ax.annotate(label, fontsize=20, xy=(tr['lon'][k_arrow] + dx_arrow, tr['lat'][k_arrow]),
                xytext=(tr['lon'][k_text] + dx_text, tr['lat'][k_text]),
                arrowprops=dict(facecolor='black', arrowstyle='->', linewidth=2),
                transform=ccrs.PlateCarree())


def fig7_along_track(trend_ag, grid_glo, tracks_alb, tracks_iera,
                     alb_track='96', iera_tracks=('7', '185'),
                     alb_extent=(-6, -1, 35, 37), iera_extent=(25, 30, 31, 35.6)):
    """Gridded all-sat-glo trends with L3-ref and L4 along-track trends.

    (a) Alboran Sea map with track ``alb_track``; (b) trend along that track;
    (c) Ierapetra map with tracks ``iera_tracks``; (d) and (e) trends along them.
    Solid lines: significant trends; dashed: all trends.
    """
    fig = plt.figure(figsize=(30, 15))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 2], hspace=0.4, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.Mercator())
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1:, 0], projection=ccrs.Mercator())
    ax5 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 1])

    # --- Alboran
    norm_a = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=5)
    ax1.set_extent(alb_extent, crs=ccrs.PlateCarree())
    ax1.add_feature(land('10m'))
    gl = _gridlines(ax1)
    gl.ylocator = mticker.FixedLocator(np.arange(35, 37.1, 0.5))
    _trend_background(ax1, grid_glo, trend_ag, norm_a)
    tr = tracks_alb[alb_track]
    sc = _track_scatter(ax1, tr, norm_a)
    ax1.scatter(tr['lon'], tr['lat'], s=15, color='black', zorder=130, transform=ccrs.PlateCarree())
    cbar = fig.colorbar(sc, ax=ax1, orientation='vertical', fraction=0.05, pad=0.25)
    cbar.set_label('Trends (cm$^2$ s$^{-2}$ year$^{-1}$)', fontsize=20)
    cbar.ax.tick_params(labelsize=20)
    _annotate_track(ax1, tr, alb_track, k_text=2, k_arrow=9)
    _trend_profile(ax2, tr, f'Track {alb_track}')
    ax2.legend(fontsize=20, ncol=2)

    # --- Ierapetra
    norm_i = TwoSlopeNorm(vmin=-5, vcenter=0, vmax=4)
    ax3.set_extent(iera_extent, crs=ccrs.PlateCarree())
    ax3.add_feature(land())
    _gridlines(ax3)
    _trend_background(ax3, grid_glo, trend_ag, norm_i)
    for t in iera_tracks:
        sc = _track_scatter(ax3, tracks_iera[t], norm_i)
        ax3.scatter(tracks_iera[t]['lon'], tracks_iera[t]['lat'], s=15, color='black', zorder=130,
                    transform=ccrs.PlateCarree())
        _annotate_track(ax3, tracks_iera[t], t, k_text=0, k_arrow=min(10, len(tracks_iera[t]['lon']) - 1),
                        dx_text=0.18)
    cbar = fig.colorbar(sc, ax=ax3, orientation='vertical', fraction=0.03, pad=0.1)
    cbar.set_label('Trends (cm$^2$ s$^{-2}$ year$^{-1}$)', fontsize=20)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_ticks(np.linspace(-5, 4, 10))
    _trend_profile(ax5, tracks_iera[iera_tracks[1]], f'Track {iera_tracks[1]}')
    _trend_profile(ax4, tracks_iera[iera_tracks[0]], f'Track {iera_tracks[0]}')
    ax4.set_xlabel('Points', fontsize=20)

    for ax, letter in zip((ax1, ax2, ax3, ax5, ax4), 'abcde'):
        ax.set_title(f'({letter})', fontsize=25, fontweight='bold')
    return fig


# --------------------------------------------------------------------------
# Figure 8: eddy statistics
# --------------------------------------------------------------------------
EDDY_STYLE = {  # key: (label, colour)
    ('AG', 'anti'): ('all-sat-glo - anticyclonic', 'darkorange'),
    ('AG', 'cycl'): ('all-sat-glo - cyclonic', 'gold'),
    ('TG', 'anti'): ('two-sat-glo - anticyclonic', 'blue'),
    ('TG', 'cycl'): ('two-sat-glo - cyclonic', 'lightsteelblue'),
}


def _draw_eddies(ax, ds, date, color):
    day = np.datetime64(date)
    sub = ds.isel(obs=np.where(ds['time'].values == day)[0])
    lons, lats = sub['effective_contour_longitude'].values, sub['effective_contour_latitude'].values
    for lo, la in zip(lons, lats):
        if np.all(np.isfinite(lo)) and np.all(np.isfinite(la)):
            ax.fill(lo, la, transform=ccrs.PlateCarree(), facecolor=color, edgecolor='black',
                    linewidth=0.4, alpha=0.7)


def fig8_eddies(eddies, stats_by_key, date='2020-01-04', extent=(-6, 6, 35, 40)):
    """Eddies of one day (a) and annual statistics (b-e) from the META3.2 DT atlas.

    Parameters
    ----------
    eddies : ``{(product, polarity): Dataset}`` with product in AG/TG, polarity in anti/cycl
    stats_by_key : ``{(product, polarity): eke_analysis.eddy_yearly_stats(...)}``
    """
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(3, 4, height_ratios=[4, 2, 2], figure=fig)
    plt.subplots_adjust(hspace=0.45)

    ax1 = fig.add_subplot(gs[0, :], projection=ccrs.Mercator())
    ax1.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#e6e6e6')
    ax1.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.8)
    gl = ax1.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)
    gl.top_labels = gl.right_labels = gl.xlines = gl.ylines = False
    gl.xlabel_style = gl.ylabel_style = {'size': 12, 'color': 'black'}
    ax1.set_extent(extent, crs=ccrs.PlateCarree())
    for key in [('AG', 'anti'), ('AG', 'cycl'), ('TG', 'anti'), ('TG', 'cycl')]:
        _draw_eddies(ax1, eddies[key], date, EDDY_STYLE[key][1])
    leg = ax1.legend([Line2D([0], [0], color=c, lw=5) for _, c in EDDY_STYLE.values()],
                     [l for l, _ in EDDY_STYLE.values()],
                     title=r'$\mathbf{a)}$   ' + np.datetime_as_string(np.datetime64(date), unit='D'),
                     fontsize=13, title_fontsize=15, loc='upper left')
    leg.get_frame().set_edgecolor('black')

    panels = [('number', 'Number of eddies', gs[1, 0:2], 'b'),
              ('radius', 'Mean radius (km)', gs[1, 2:4], 'c'),
              ('area', r'Total area ($\times$10$^8$ km$^2$)', gs[2, 0:2], 'd'),
              ('speed', 'Mean speed (cm s$^{-1}$)', gs[2, 2:4], 'e')]
    for name, title, spec, letter in panels:
        ax = fig.add_subplot(spec)
        for key, (label, color) in EDDY_STYLE.items():
            years, values, reg = stats_by_key[key][name]
            ax.plot(years, values, color=color, lw=3, label=label)
            ax.plot(years, reg.slope * years + reg.intercept, color=color, ls='--')
        ax.tick_params(axis='both', labelsize=13)
        ax.text(0.02, 1.02, f'({letter})', transform=ax.transAxes, fontsize=18, fontweight='bold')
        ax.set_title(title, fontsize=15)
        ax.grid(alpha=0.3)
        if letter in 'de':
            ax.set_xlabel('Time', fontsize=15)
    return fig
