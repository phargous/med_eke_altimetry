"""
Analysis helpers for the figures of

    Hargous, Combes, Barcelo-Llull, Pascual (2026), "Eddy kinetic energy
    variability from 30 years of altimetry in the Mediterranean Sea",
    Ocean Science 22, 549-564.

Everything that is computation (region masks, area-weighted time series, trend
tests, satellite-count correction, along-track L3/L4 trends, eddy-atlas
statistics) lives here. The plotting functions are in ``eke_figures.py`` and
the notebook ``notebooks/Paper_figures.ipynb`` strings the two together.

Conventions
-----------
* EKE is in cm2 s-2, trends are in cm2 s-2 yr-1.
* Gridded EKE has dimensions ``(time, lat, lon)``; masks are boolean
  ``(lat, lon)`` arrays defined on the grid of the product they are applied to.
* Product keys: ``AM`` = all-sat-euro (1/8 deg), ``AG`` = all-sat-glo (1/4 deg),
  ``TG`` = two-sat-glo (1/4 deg).
"""

from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.path import Path as MplPath
from scipy import stats

import pymk  # local copy of pymannkendall (also returns the autocorrelation factor n_ns)
import utils  # haversine, get_area_haversine

ALPHA = 0.05  # significance level used everywhere in the paper

# --------------------------------------------------------------------------
# Regions
# --------------------------------------------------------------------------
# Mediterranean polygon (lon, lat); only used to select the META3.2 eddies that
# lie inside the Mediterranean Sea (see ``merge_med_eddies``).
MED_POLYGON = [(-6, 35), (19, 29.5), (36, 30), (37, 37.5), (13.5, 46.5), (0, 43), (-6, 36.5)]

REGION_NAMES = {'MED': 'Mediterranean Sea', 'ALB': 'Alboran Sea',
                'ALBB': 'Eastern Alboran', 'IERA': 'Ierapetra area'}

# Pre-computed boolean masks (``results/masks/``), one per region and grid.
# Ierapetra = isoline of 200 cm2 s-2 of the 30-year mean all-sat-euro EKE.
MASK_FILES = {
    'MED':  {'med': '2mask_MED_MED.npy',     'glo': '2mask_MED_GLO.npy'},
    'ALB':  {'med': '2mask_ALB_MED.npy',     'glo': '2mask_ALB_GLO.npy'},
    'ALBB': {'med': '2mask_ALBB_MED.npy',    'glo': '2mask_ALBB_GLO.npy'},
    'IERA': {'med': '2mask_IERAISO_MED.npy', 'glo': '2mask_IERAISO_GLO.npy'},
}


def load_region_masks(mask_dir, grid_med=None, grid_glo=None):
    """Load the pre-computed region masks.

    Returns ``{region: {'med': mask_on_1/8deg_grid, 'glo': mask_on_1/4deg_grid}}``
    (boolean ``(lat, lon)`` arrays). If the grids are given, the shapes are checked.
    """
    mask_dir = Path(mask_dir)
    masks = {}
    for region, files in MASK_FILES.items():
        masks[region] = {g: np.load(mask_dir / f, allow_pickle=True).astype(bool)
                         for g, f in files.items()}
    for g, grid in (('med', grid_med), ('glo', grid_glo)):
        if grid is None:
            continue
        shape = (grid.sizes['lat'], grid.sizes['lon'])
        for region in masks:
            if masks[region][g].shape != shape:
                raise ValueError(f"Mask {region}/{g} has shape {masks[region][g].shape}, "
                                 f"expected {shape}")
    return masks


# --------------------------------------------------------------------------
# Area-weighted time series
# --------------------------------------------------------------------------
def area_weighted_mean(da, mask=None, chunk=1000):
    """Area-weighted spatial mean of ``da`` (time, lat, lon) over ``mask``.

    Cell areas come from ``utils.get_area_haversine``. Only cells that hold
    data at least once are used in the normalisation. Data are read by time
    chunks so that large files do not have to fit in memory twice.
    """
    area = utils.get_area_haversine(da.lat.values, da.lon.values)
    keep = np.isfinite(area)
    if mask is not None:
        keep &= np.asarray(mask, dtype=bool)
    weights = np.where(keep, area, 0.0)

    nt = da.sizes['time']
    valid = np.zeros(area.shape, dtype=bool)
    total = np.empty(nt)
    for i in range(0, nt, chunk):
        x = da.isel(time=slice(i, i + chunk)).values.astype(np.float64)
        x = np.where(keep, x, np.nan)
        valid |= np.isfinite(x).any(axis=0)
        total[i:i + chunk] = np.nansum(x * weights, axis=(1, 2))
    total /= weights[valid].sum()
    return xr.DataArray(total, dims='time', coords={'time': da.time.values}, name=da.name)


def regional_series(eke, masks, regions=('MED', 'ALB', 'ALBB', 'IERA')):
    """Area-weighted EKE series ``{region: {'AM', 'AG', 'TG'}}``.

    Parameters
    ----------
    eke : dict
        ``{'AM': DataArray, 'AG': DataArray, 'TG': DataArray}`` of EKE (time, lat, lon).
    masks : dict
        Output of :func:`load_region_masks`.
    """
    grid = {'AM': 'med', 'AG': 'glo', 'TG': 'glo'}
    return {r: {p: area_weighted_mean(eke[p], masks[r][grid[p]]) for p in eke}
            for r in regions}


# --------------------------------------------------------------------------
# Trends
# --------------------------------------------------------------------------
@dataclass
class Trend:
    """Yue-Wang-corrected Theil-Sen trend of a regular time series."""
    slope: float      # per year
    p: float
    n_ns: float       # variance inflation factor due to autocorrelation
    intercept: float  # intercept of the Theil-Sen line (index 0 = first sample)
    std: float        # standard error of the slope (per year), effective sample size
    n: int
    samples_per_year: float = 365.0

    @property
    def significant(self):
        return self.p <= ALPHA

    def line(self):
        """Fitted straight line evaluated at every sample."""
        return self.slope / self.samples_per_year * np.arange(self.n) + self.intercept


def compute_trend(ts, samples_per_year=365.0):
    """Trend of a 1-D series with the Yue and Wang (2004) modified Mann-Kendall test.

    The standard error uses the effective sample size ``n / n_ns``:
    ``std = spy * sqrt(sum(res**2) / (ess - 2)) / sqrt(sum((x - mean(x))**2))``.
    """
    ts = np.asarray(ts, dtype=float)
    if np.isnan(ts).all():
        return Trend(np.nan, np.nan, np.nan, np.nan, np.nan, ts.size, samples_per_year)
    r = pymk.yue_wang_modification_test(ts)
    slope = r.slope * samples_per_year
    x = np.arange(ts.size)
    resid = ts - (r.slope * x + r.intercept)
    ess = ts.size / r.n_ns
    std = samples_per_year * np.sqrt(np.nansum(resid ** 2) / (ess - 2)) \
        / np.sqrt(np.sum((x - x.mean()) ** 2))
    return Trend(slope, r.p, r.n_ns, r.intercept, std, ts.size, samples_per_year)


# --------------------------------------------------------------------------
# Satellite constellation
# --------------------------------------------------------------------------
def satellite_timeline(missions, start='1993-01-01', end='2023-06-07',
                       last_known='2023-01-01', n_after=7):
    """Daily number of altimeters used by the all-sat products.

    Parameters
    ----------
    missions : dict
        ``{mission: (start_date, end_date)}`` active periods.
    last_known : str
        End of the mission table; after this date the count is held at ``n_after``
        (7 satellites between January and June 2023).
    """
    t = pd.DataFrame(index=pd.date_range(start, last_known, freq='D'), data={'n': 0})
    for t0, t1 in missions.values():
        t.loc[t0:t1, 'n'] += 1
    tail = pd.DataFrame(index=pd.date_range(pd.Timestamp(last_known) + pd.Timedelta('1D'), end, freq='D'),
                        data={'n': n_after})
    return pd.concat([t, tail])['n']


def correct_for_satellite_count(all_sat, two_sat, n_sat, degree=2):
    """Remove the part of all-sat EKE that depends on the number of satellites.

    A polynomial of degree ``degree`` is fitted to ``all_sat - two_sat`` as a
    function of the number of satellites and subtracted from ``all_sat``
    (``all-sat-glo-corrected`` in the paper).

    Returns
    -------
    corrected : numpy.ndarray
    coefficients : numpy.ndarray (highest degree first)
    """
    diff = np.asarray(all_sat) - np.asarray(two_sat)
    n = np.asarray(n_sat, dtype=float)
    coef = np.polyfit(n, diff, degree)
    return np.asarray(all_sat) - np.polyval(coef, n), coef


def unweighted_masked_mean(da, mask):
    """Plain (non area-weighted) spatial mean over ``mask``.

    The satellite-count correction of the paper was fitted on these
    unweighted means; the difference with the weighted mean is small.
    """
    return da.where(xr.DataArray(np.asarray(mask), dims=('lat', 'lon'),
                                 coords={'lat': da.lat, 'lon': da.lon})).mean(['lat', 'lon'])


# --------------------------------------------------------------------------
# Gridded trends and correlations
# --------------------------------------------------------------------------
def _pixel_trend(ts):
    t = compute_trend(ts)
    return t.slope, t.p, t.n_ns, t.intercept


def trend_map(da, n_jobs=1):
    """Yue-Wang trend at every grid point.

    Warning: the Theil-Sen estimator and the autocorrelation are O(n**2) in the
    series length: roughly 2-10 s per grid point for 30 years of daily data
    (about 1 h at 1/4 deg and several hours at 1/8 deg for the Mediterranean).
    Use ``n_jobs > 1`` (joblib, about 1 GB of memory per worker) and cache the
    result (see :func:`cached`).

    Returns
    -------
    dict with ``slope`` (cm2 s-2 yr-1), ``pvalue``, ``nns`` (autocorrelation
    factor) and ``intercept`` arrays of shape (lat, lon).
    """
    arr = da.transpose('time', 'lat', 'lon').values
    nlat, nlon = arr.shape[1:]
    out = {k: np.full((nlat, nlon), np.nan) for k in ('slope', 'pvalue', 'nns', 'intercept')}
    pixels = [(i, j) for i in range(nlat) for j in range(nlon) if np.isfinite(arr[:, i, j]).any()]
    if n_jobs == 1:
        results = [_pixel_trend(arr[:, i, j]) for i, j in pixels]
    else:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs)(delayed(_pixel_trend)(arr[:, i, j]) for i, j in pixels)
    for (i, j), (s, p, n, c) in zip(pixels, results):
        out['slope'][i, j], out['pvalue'][i, j], out['nns'][i, j], out['intercept'][i, j] = s, p, n, c
    return out


def load_legacy_trend_dict(path):
    """Read a ``Redo/*_slpintns.npy`` dictionary written by the original scripts.

    Those stored slopes per *day*; this converts them to cm2 s-2 yr-1.
    """
    d = np.load(path, allow_pickle=True).item()
    d = dict(d)
    d['slope'] = d['slope'] * 365.0
    return d


def pearson_map(x, y):
    """Pearson correlation between a 1-D series ``x`` and every column of ``y``.

    Equivalent to calling ``scipy.stats.pearsonr`` on every grid point after
    dropping the time steps where either series is NaN, but vectorised.

    Parameters
    ----------
    x : (time,) array
    y : (time, lat, lon) array

    Returns
    -------
    r, p : (lat, lon) arrays (NaN where fewer than 3 valid samples).
    """
    x = np.asarray(x, dtype=float)[:, None, None]
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y) & np.isfinite(x)
    n = ok.sum(0)
    xs = np.where(ok, x, 0.0)
    ys = np.where(ok, y, 0.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        mx = xs.sum(0) / n
        my = ys.sum(0) / n
        dx = np.where(ok, x - mx, 0.0)
        dy = np.where(ok, y - my, 0.0)
        r = (dx * dy).sum(0) / np.sqrt((dx ** 2).sum(0) * (dy ** 2).sum(0))
        r = np.clip(r, -1, 1)
        t = r * np.sqrt((n - 2) / (1 - r ** 2))
        p = 2 * stats.t.sf(np.abs(t), n - 2)
    bad = n <= 2
    return np.where(bad, np.nan, r), np.where(bad, np.nan, p)


def cached(path, func, *args, **kwargs):
    """Return ``np.load(path)`` if it exists, otherwise compute with ``func`` and save."""
    path = Path(path)
    if path.exists():
        return np.load(path, allow_pickle=True).item()
    result = func(*args, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, result)
    return result


def trend_table(trend_maps, series_trends):
    """Content of Table 1 for one EKE definition.

    Parameters
    ----------
    trend_maps : dict
        ``{product: output of trend_map}``
    series_trends : dict
        ``{product: Trend}`` of the area-weighted Mediterranean mean series.

    Returns
    -------
    pandas.DataFrame (rows: statistic, columns: product), cm2 s-2 yr-1.
    """
    rows = {'Mean of all trends': {}, 'Mean of the significant trends': {}, 'Trend of the mean': {}}
    for p, m in trend_maps.items():
        s, pv = m['slope'], m['pvalue']
        rows['Mean of all trends'][p] = np.nanmean(s)
        rows['Mean of the significant trends'][p] = np.nanmean(np.where(pv <= ALPHA, s, np.nan))
        rows['Trend of the mean'][p] = series_trends[p].slope
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------
# Along-track (L3) versus gridded (L4) trends  -  Fig. 7
# --------------------------------------------------------------------------
G = 9.81
OMEGA = 2 * np.pi / (23 * 3600 + 56 * 60 + 4)  # Earth rotation rate (sidereal day)


def across_track_eke(sla, lon, lat):
    """EKE_perp (cm2 s-2) from along-track sea level anomalies.

    The across-track geostrophic velocity is ``v = g / f * d(sla)/ds`` with a
    5-point centred finite difference in the interior and one-sided stencils at
    the four extreme points (as in Pujol et al. type L3 processing). Rotating
    ``v`` into (u, v) components does not change its modulus, so
    ``EKE_perp = 0.5 * v**2``.

    Parameters
    ----------
    sla : (npoints, ntime) array, metres, points ordered along the track
    lon, lat : (npoints,) arrays, degrees
    """
    sla = np.asarray(sla, dtype=float)
    n = sla.shape[0]
    f = 2 * OMEGA * np.sin(np.deg2rad(np.asarray(lat)))[:, None]
    d = utils.haversine(lon[:-1], lat[:-1], lon[1:], lat[1:]) * 1000.0  # spacing between neighbours, m
    dsig = np.empty(n)
    dsig[1:-1] = utils.haversine(lon[:-2], lat[:-2], lon[2:], lat[2:]) * 1000.0  # i-1 -> i+1
    dsig[0], dsig[-1] = d[0], d[-1]
    ds = np.empty((n, 1))
    ds[:, 0] = dsig

    deriv = np.empty_like(sla)
    deriv[2:-2] = (-sla[4:] + 8 * sla[3:-1] - 8 * sla[1:-3] + sla[:-4]) / (6 * ds[2:-2])
    deriv[0] = (-3 * sla[0] + 4 * sla[1] - sla[2]) / (2 * ds[0])
    deriv[1] = (-2 * sla[0] - 3 * sla[1] + 6 * sla[2] - sla[3]) / (6 * d[0])
    deriv[-1] = (sla[-3] - 4 * sla[-2] + 3 * sla[-1]) / (2 * ds[-1])
    deriv[-2] = (sla[-4] - 6 * sla[-3] + 3 * sla[-2] + 2 * sla[-1]) / (6 * d[-2])
    v = G / f * deriv * 100.0  # cm/s
    return 0.5 * v ** 2


def open_l3(files, bbox):
    """Open L3 along-track files, keep the box ``bbox = (lonmin, lonmax, latmin, latmax)``.

    Returns a DataFrame with columns time, lon, lat, track, sla (metres).
    """
    lonmin, lonmax, latmin, latmax = bbox
    frames = []
    for f in files:
        ds = xr.open_dataset(f)
        lon = ds['longitude'].values
        lon = np.where(lon > 180, lon - 360, lon)
        lat = ds['latitude'].values
        keep = (lon >= lonmin) & (lon <= lonmax) & (lat >= latmin) & (lat <= latmax)
        frames.append(pd.DataFrame({'time': ds['time'].values[keep], 'lon': lon[keep], 'lat': lat[keep],
                                    'track': ds['track'].values[keep],
                                    'sla': ds['sla_filtered'].values[keep]}))
        ds.close()
    df = pd.concat(frames, ignore_index=True).sort_values('time')
    df['time'] = df['time'].dt.floor('D')
    return df


def open_l4_sla(directory, bbox, var='sla'):
    """Open yearly L4 files stored as ``directory/<year>/<file>.nc`` and crop to ``bbox``.

    Missing values are linearly interpolated (and extrapolated) along time.
    """
    lonmin, lonmax, latmin, latmax = bbox
    parts = []
    for year_dir in sorted(p for p in Path(directory).iterdir() if p.is_dir() and not p.name.startswith('.')):
        f = sorted(q for q in year_dir.iterdir() if q.suffix == '.nc' and not q.name.startswith('.'))[0]
        ds = xr.open_dataset(f)[[var]].rename({'longitude': 'lon', 'latitude': 'lat'})
        ds = ds.assign_coords(lon=xr.where(ds.lon > 180, ds.lon - 360, ds.lon))
        ds = ds.sel(lon=slice(lonmin, lonmax), lat=slice(latmin, latmax))
        parts.append(ds.load())
    da = xr.concat(parts, dim='time')[var]
    da['time'] = da['time'].dt.round('D')
    return da.interpolate_na(dim='time', method='linear', fill_value='extrapolate')


def _sample_l4_at_l3_times(l4, lon, lat, times):
    """L4 SLA at one L3 point and at the L3 sampling dates (linear, nearest as fallback)."""
    lin = l4.interp(lon=lon, lat=lat, method='linear')
    near = l4.interp(lon=lon, lat=lat, method='nearest')
    s = lin.where(lin.notnull(), near)
    s = s.interp(time=times)
    return s.interpolate_na(dim='time', method='linear').values


def along_track_trends(l3_files, l4_dirs, bbox, min_samples=800, step_days=10):
    """EKE_perp trends along reference tracks for L3 and several L4 products (Fig. 7b, d, e).

    For every point of every reference track:

    1. the L3 SLA is linearly interpolated on a common regular grid
       (``step_days``) shared by all points of the track;
    2. each L4 product is sampled at the position of the point and at the
       dates of the L3 measurements, then interpolated on the same grid, so
       that L3 and L4 are compared with the same temporal sampling;
    3. EKE_perp is computed along the track (:func:`across_track_eke`) and its
       Yue-Wang trend estimated.

    Parameters
    ----------
    l3_files : list of paths
    l4_dirs : dict ``{name: directory}`` (e.g. ``{'AM': ..., 'AG': ..., 'TG': ...}``)
    bbox : (lonmin, lonmax, latmin, latmax)
    min_samples : points with fewer L3 samples are discarded (< 1 per 14 days)

    Returns
    -------
    dict ``{track: {'lon', 'lat', 'L3': {'slope', 'p'}, <name>: {'slope', 'p'}}}``
    with slopes in cm2 s-2 yr-1.
    """
    spy = 365.0 / step_days
    l3 = open_l3(l3_files, bbox)
    l4 = {k: open_l4_sla(d, bbox) for k, d in l4_dirs.items()}

    out = {}
    for track, dft in l3.groupby('track'):
        pts = []
        for (lon, lat), g in dft.groupby(['lon', 'lat']):  # sorted by lon then lat
            g = g.drop_duplicates('time')
            if len(g) > min_samples:
                pts.append((lon, lat, g['time'].values, g['sla'].values))
        if len(pts) <= 4:
            continue
        grid = pd.date_range(max(p[2][0] for p in pts), min(p[2][-1] for p in pts), freq=f'{step_days}D')
        tnum = grid.values.astype('datetime64[ns]').astype(float)

        def on_grid(times, values):
            return np.interp(tnum, times.astype('datetime64[ns]').astype(float), values)

        lon = np.array([p[0] for p in pts])
        lat = np.array([p[1] for p in pts])
        series = {'L3': np.array([on_grid(p[2], p[3]) for p in pts])}
        for k, field in l4.items():
            series[k] = np.array([on_grid(p[2], _sample_l4_at_l3_times(field, p[0], p[1], p[2]))
                                  for p in pts])

        res = {'lon': lon, 'lat': lat}
        for k, sla in series.items():
            eke = across_track_eke(sla, lon, lat)
            tr = [compute_trend(e, samples_per_year=spy) for e in eke]
            res[k] = {'slope': np.array([t.slope for t in tr]), 'p': np.array([t.p for t in tr])}
        out[str(int(track))] = res
    return out


# --------------------------------------------------------------------------
# Eddy atlas (META3.2 DT)  -  Fig. 8
# --------------------------------------------------------------------------
EDDY_VARS = ['longitude', 'latitude', 'time', 'effective_radius', 'effective_area', 'speed_average',
             'effective_contour_longitude', 'effective_contour_latitude']


def merge_med_eddies(src_dir, out_dir, polarity):
    """Extract the Mediterranean eddies from the global META3.2 DT atlas.

    ``polarity`` is ``'Cyclonic'`` or ``'Anticyclonic'``. The ``long``,
    ``short`` and ``untracked`` files are concatenated (no lifetime threshold
    is applied, as in the paper) and the result saved as
    ``Merged_Med_{cycl|anti}_eddies.nc`` in ``out_dir``.
    The longitudes are wrapped to [-180, 180].
    """
    out = Path(out_dir) / f"Merged_Med_{'cycl' if polarity == 'Cyclonic' else 'anti'}_eddies.nc"
    if out.exists():
        return out
    poly = MplPath(MED_POLYGON)
    parts = []
    for kind in ('long', 'short', 'untracked'):
        f = sorted(Path(src_dir).glob(f'*_{polarity}_{kind}_*.nc'))[0]
        ds = xr.open_dataset(f)
        for v in ('longitude', 'effective_contour_longitude', 'longitude_max', 'speed_contour_longitude'):
            if v in ds:
                ds[v] = xr.where(ds[v] > 180, ds[v] - 360, ds[v])
        inside = poly.contains_points(np.column_stack([ds['longitude'].values, ds['latitude'].values]))
        keep = [v for v in EDDY_VARS if v in ds]
        parts.append(ds[keep].isel(obs=inside))
    out.parent.mkdir(parents=True, exist_ok=True)
    xr.concat(parts, dim='obs').to_netcdf(out)
    return out


def get_med_eddies(src_dir, cache_dir, polarity):
    """Path of the merged Mediterranean eddy file, building it in ``cache_dir`` if needed.

    ``src_dir`` (raw data) is never written to: an existing merged file found
    there is reused, otherwise it is created in ``cache_dir``.
    """
    name = f"Merged_Med_{'cycl' if polarity == 'Cyclonic' else 'anti'}_eddies.nc"
    if (Path(src_dir) / name).exists():
        return Path(src_dir) / name
    return merge_med_eddies(src_dir, cache_dir, polarity)


def yearly(values, times, how='mean'):
    """Annual ``mean``, ``sum`` or ``count`` of ``values`` (year-end labels)."""
    s = pd.Series(values, index=pd.to_datetime(times))
    r = s.resample('YE')
    return {'mean': r.mean, 'sum': r.sum, 'count': r.count}[how]()


def eddy_yearly_stats(ds, last_date='2021-12-31'):
    """Annual number, mean radius (km), total area (1e8 km2) and mean speed (cm/s) of the eddies in ``ds``.

    Returns ``{name: (years, values, linregress_result)}``.
    """
    ds = ds.sel(obs=ds['time'] <= np.datetime64(last_date))
    t = ds['time'].values
    series = {
        'number': yearly(np.ones(t.size), t, 'sum'),
        'radius': yearly(ds['effective_radius'].values / 1e3, t),
        'area': yearly(ds['effective_area'].values / 1e14, t, 'sum'),
        'speed': yearly(ds['speed_average'].values * 100, t),
    }
    return {k: (v.index.year.values, v.values, stats.linregress(v.index.year, v.values))
            for k, v in series.items()}
