# Figure forensics: how the flagship PM2.5 papers make their figures

Compiled 2026-08-08 from repo audits, PDF raster extraction, and colorbar
pixel measurement. Purpose: replicate the field's figures exactly, then
improve them for AQNet. Full agent evidence in the session archive; hex
values below are measured from the published artifacts.

## Software by group (evidence-based, not assumed)

| Group / product | Maps made with | Evidence |
|---|---|---|
| Wei GHAP (Nat Commun 2023) | ArcGIS Pro 3.0.1 | stated in every map caption; Excel + Python for charts |
| van Donkelaar / ACAG | MATLAB | their own import/plot tutorial; GBD-MAPS release is 98% MATLAB |
| Di / Harvard 2019 | base R | ramps are R topo.colors() and cm.colors(); GAM plots; image.plot layout |
| EPA Downscaler | MATLAB (model), R (report figures) | EPA/600/C-12/002 manual; lattice + cut() interval legends |
| ML-era papers (DeepAir, TransNet, Aurora, IGNNK) | matplotlib (+cartopy/seaborn/geopandas) | repo code read directly |

Cartopy appears in none of the flagship product releases; it is the
ML-paper stack. Nobody in the audited set uses ggplot for maps.

## Measured colormaps (hex stops, low to high)

* Wei GHAP global map == matplotlib RdYlBu_r (RMS 5.3/255). Sampled:
  292e96 406baf 669cc8 92c4dd bde1ef e5f6ee fefdbd fee797 fcbe71 fa9052
  eb5c3c cf2a28 a50527. Range 8-72 ug/m3, plate carree, white ocean,
  black country + gray state lines, red inset rectangles.
* ACAG V5 North America == same RdYlBu_r family (RMS 12.9, JPEG-limited).
  Sampled: 3b4084 466faf 68a2c8 90c4da bfe1ed ebf6f2 ffffbe f4e195
  fabd6d ee8448 e35434 c52a28 8d1226. Global maps on a quasi-log tick
  set (1, 5, 10, 20, 30, 50, 80).
* Di 2019 == R topo.colors() (blue -> cyan -> green -> yellow -> tan):
  210897 000aff 0460ff 03bdf6 00ff29 41ff00 9eff00 fffa08 fde442 f9db83
  f6e3b9. Uncertainty maps: R cm.colors() cyan-white-magenta.
* EPA DS reports: discrete 8-9 bins gray/lightblue/blue/green/yellow/
  orange/red/darkred/magenta with lattice gray strip headers.
* DeepAir: jet-family density scatters; RdYlGn_r-style CA maps 0-30;
  Reds station-R2 dots on satellite imagery.
* Consensus in ML papers: RdBu_r for model-minus-truth difference maps;
  RdYlGn_r for station values; dpi 300-600; bold size-16 labels.

## Scatter-plot conventions (the canonical evaluation panel)

2D-histogram density scatter (RdYlBu_r or jet family), black dashed 1:1,
thin red least-squares line, stats block upper-left in the form:
Y = 0.89X + 2.89 / R2 / RMSE / (NRMSE, MAE optional), N. Harvard instead
uses GAM spline fits with 95% CI dashed lines.

## Exact-replication routes (fastest first)

1. Wei-style map: matplotlib, cmap RdYlBu_r, BoundaryNorm on their tick
   step, white ocean, TIGER/Natural Earth boundaries. Pixel-faithful with
   the measured stops.
2. ACAG map: same with the quasi-log tick set; their data is public
   (s3://satpmdata, no-sign-request) for a literal regeneration.
3. Di map: matplotlib clone of topo.colors (stops above) on their SEDAC
   GeoTIFFs, or run R fields::image.plot directly.
4. EPA FAQSD: tract CSVs from RSIG + pandas.cut into their bins.
5. DeepAir: their repo runs for scatters/violins; the CA map script was
   never committed (rewrite is ~30 lines).

## Fingerprinting toolkit (for any future paper)

PDF: pdffonts per figure page (DejaVuSans subset => matplotlib;
Helvetica => R/MATLAB; Computer Modern => LaTeX); pdfimages -list/-png
for lossless rasters; pdftocairo -svg turns a vector colorbar into
literal hex fills. PNG: matplotlib Agg stamps tEXt Software chunk.
Raster ramps: sample 256 points along the colorbar, median across width,
match against all registered matplotlib/cmocean/colorcet/crameri maps in
CIELAB; deltaE < 3 means a named map, else keep sampled anchors.
Heatmap-to-data inversion: KDTree in Lab space over the extracted ramp,
rescaled by the colorbar ticks.

## AQNet house decisions informed by this

* Our EPA-anchored blue-white-orange diverging scale stays the default
  for policy-facing maps (none of the above encode a health standard;
  that is our improvement).
* For like-for-like comparison figures against Wei/ACAG products (A15,
  A18), render BOTH sides in measured RdYlBu_r with their conventions so
  the comparison is style-neutral.
* Adopt their density-scatter stats block (with slope) in F23-style
  panels; keep protocol labels in the first caption sentence.
