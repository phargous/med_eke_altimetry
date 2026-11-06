# Eddy kinetic energy variability in the Mediterranean Sea — figure reproduction

Code to reproduce the figures and Table 1 of

> Hargous, P., Combes, V., Barceló-Llull, B., Pascual, A. (2026).
> *Eddy kinetic energy variability from 30 years of altimetry in the Mediterranean Sea.*
> Ocean Science, 22, 549–564. https://os.copernicus.org/articles/22/549/2026/

The paper asks whether the eddy kinetic energy (EKE) of the Mediterranean Sea has increased
over 1993–2023 in CMEMS DUACS L4 altimetry, and how much of the trend comes from the changing
satellite constellation rather than from the ocean. The notebook rebuilds every figure from the
EKE fields, the pre-computed region masks and the raw L3/L4/eddy-atlas files.

## Repository layout

```
code/
├── eke_analysis.py     # computations: area-weighted series, Yue–Wang trends, satellite-count
│                       # correction, gridded and along-track trends, eddy statistics
├── eke_figures.py      # one plotting function per figure
├── pymk.py             # local copy of pymannkendall (modified Mann–Kendall tests)
├── utils.py            # haversine distances and cell areas
└── notebooks/
    └── Paper_figures.ipynb   # the notebook: Figures 1–8 and Table 1
```

## Data

The data are **not** included. The notebook expects:

| Item                                                         | Location                                                                                | Source                                                                                               |
| ------------------------------------------------------------ | --------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| EKE fields (`NEKE_*`, `EKE_*`), 1993-01-01 to 2023-06-07 | `processed_data/{MED,GLO}/{ALL_SAT,TWO_SAT}/`                                         | computed from CMEMS L4 SLA (see below)                                                               |
| Region masks                                                 | `results/masks/2mask_{MED,ALB,ALBB,IERAISO}_{MED,GLO}.npy`                            | computed beforehand (Ierapetra = 200 cm² s⁻² isoline of the 30-year mean EKE of `all-sat-euro`) |
| L4 altimetry                                                 | `data/{med_allsat_L4_008_068, glo_allsat_L4_008_047, glo_twosat_L4_008_057}/vDT2021/` | [CMEMS](https://marine.copernicus.eu)                                                                 |
| L3 along-track altimetry                                     | `data/eur_L3_008_061/vDT2021/processed/`                                              | CMEMS                                                                                                |
| Satellite periods and track files                            | `data/satellites.npy`, `data/20days/*.nc`                                           | CMEMS                                                                                                |
| Eddy atlas META3.2 DT                                        | `data/eddy_aviso/META3.2_DT_{allsat,twosat}/`                                         | AVISO                                                                                                |

The three compared products are `all-sat-euro` (1/8°), `all-sat-glo` (1/4°) and `two-sat-glo`
(1/4°, fixed two-satellite constellation). EKE is computed from the geostrophic velocity
anomalies with respect to the 1993–2023 mean.

### Computation time

The slow steps are cached in `results/paper_cache/`, so they only run once:

* **Trend maps (Fig. 6, Table 1)**: the Yue–Wang test with Theil–Sen slope costs about 2–10 s
  per grid point; expect hours at 1/8°. Use `N_JOBS` (about 1 GB of memory per worker).
* **Along-track trends (Fig. 7)**: L4 fields are interpolated at every L3 point and a trend is computed for each of them.

## Citation

If you use this code, please cite the paper above.

## Acknowledgements

`pymk.py` is adapted from [pymannkendall](https://github.com/mmhs013/pyMannKendall) (MIT licence).
