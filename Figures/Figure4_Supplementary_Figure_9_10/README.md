# Figure 4 d-f and Supplementary Figures 9-10

This package generates the trajectory panels from three-dimensional
lipid-droplet localizations. Each panel has an `inputs.csv`, a runnable
`plot.py`, and one annotated `*_with_text.pdf`.

## Reproduce

Create a separate Python 3.9 environment, then install and run from this
directory:

```sh
conda create -n depict-tracking python=3.9
conda activate depict-tracking
python -m pip install -r requirements.txt
python -B rebuild.py
```

The pinned LapTrack version uses scikit-image 0.19.3. Keep this environment
separate from volumetric postprocessing, which uses scikit-image 0.24.0.

The default command links the localizations, computes motion statistics,
and exports all nine panels.

To redraw existing analysis results:

```sh
python rebuild.py --stage plot
python rebuild.py --stage plot --panels e 10b 10c
python Supplementary_Figure_9/panel_a/plot.py
```

Run these commands from this directory in the same Python environment. `--stage track` runs linking and identity association;
`--stage analyze` recomputes statistics. `scripts/` contains the shared analysis
and rendering functions; panel entry points call those functions.

Use Python 3.9 and the dependencies in `requirements.txt`. To install them in
a new environment, run `python -m pip install -r requirements.txt`.

## Panel Inputs And Outputs

All panels read the same trajectory table,
`analysis/tracks_by_resolution.csv`, or statistics computed from it. Input paths
in each panel's `inputs.csv` are relative to that panel directory.

| Panel | Directory | Content | Shared implementation |
| --- | --- | --- | --- |
| Figure 4 d | `panel_d` | 3 x 5 representative baseline trajectories | `scripts/plot_trajectory_3x5_3dbox.py` |
| Figure 4 e | `panel_e` | Speeds at four reconstruction rates, by motion class | `scripts/temporal_plots.py` |
| Figure 4 f | `panel_f` | Baseline alpha density, Gaussian components and class strip | `scripts/plot_alpha_density_motion_strip.py` |
| Supplementary 9 a | `Supplementary_Figure_9/panel_a` | All 18 confined trajectories | `scripts/plot_final_motion_trajectories.py` |
| Supplementary 9 b | `Supplementary_Figure_9/panel_b` | All 35 diffusing trajectories | `scripts/plot_final_motion_trajectories.py` |
| Supplementary 9 c | `Supplementary_Figure_9/panel_c` | All 30 directional trajectories | `scripts/plot_final_motion_trajectories.py` |
| Supplementary 10 a | `Supplementary_Figure_10/panel_a` | Six paired examples at 16.25 and 0.8125 vps | `scripts/temporal_plots.py` |
| Supplementary 10 b | `Supplementary_Figure_10/panel_b` | Four-resolution alpha distributions | `scripts/plot_alpha_temporal_resolutions.py` |
| Supplementary 10 c | `Supplementary_Figure_10/panel_c` | All-track speed distributions | `scripts/plot_all_tracks_speed_by_resolution.py` |

Panel d uses a deterministic selection of 15 representative tracks and labels
their computed mean speeds. The distributions and complete trajectory sheets
use the shared data throughout.

## Localization Data And Cohorts

`inputs/angle{12,24,48,240}/localizations.csv` contains measured candidate
positions before linking. Coordinates `z/y/x` are in voxels, and `z_um/y_um/x_um`
are in micrometers, using 0.09 um isotropic voxels. Detection IDs, voxel counts
and localization metadata are included. At 240 angles, the candidate pool
includes strong/weak localization priority and threshold columns.

Each reconstruction rate uses its own measurements. The 12-angle reference
provides droplet identities for matching the other rates; its coordinates are
not substituted for newly measured positions.

| Angles | Rate (vps) | Global frames, inclusive | Positions per complete track | Tracks |
| ---: | ---: | --- | ---: | ---: |
| 12 | 16.25 | 20-158 | 139 | 83 |
| 24 | 8.125 | 10-78 | 69 | 83 |
| 48 | 4.0625 | 5-38 | 34 | 83 |
| 240 | 0.8125 | 1-6 | 6 | 80 |

The 24-, 48- and 240-angle `frames.csv` files define exposure intervals and
inclusion flags. An angle group G and global frame i correspond to observations
`[G*i, G*(i+1))` at 195 observations/s. Time is the exposure midpoint relative
to the first baseline midpoint: `(G*(i+0.5)-246)/195` seconds. Equal global
frame numbers at different rates do not represent the same time.

Baseline linking uses a LapTrack squared-distance cutoff for a
5 um/s speed limit, gap maximum frame difference 2, and no splitting or merging.
Six uniquely matched, shared-frame fragment pairs are joined with
voxel-count-weighted measured centers. Only trajectories with measured positions
at every baseline frame are retained; there is no gap interpolation.

At each 24-, 48- or 240-angle exposure, the mean baseline position inside that exposure
provides an identity prediction. A one-to-one assignment uses a 0.30 um gate;
strong candidates have priority at 240 angles. LapTrack then verifies each
assigned identity with the same physical speed and gap limits. The full-field
native LapTrack outputs and association audits remain in `analysis/angle*/`.

Main distributions retain each rate's complete cohort independently. Their class
counts are 35/30/18 (diffusing/directional/confined) at 12, 24 and 48 angles, and
34/28/18 at 240 angles. The intersection of 80 identities is used only to choose
paired examples. Figure 4 f and Supplementary 9 use all 83 baseline identities.

## Motion Analysis

Mean speed is the mean successive three-dimensional displacement divided by the
actual midpoint time difference, converted to um/min. TA-MSD uses actual frame
lags, and equal-weight log OLS fits `ln(MSD)=ln(6*D_alpha)+alpha*ln(tau)`.
Baseline fits use lags 1-10, at least five displacement pairs per lag, and at
least four valid lags. Other rates request lags 1-4, at least three pairs per
lag, and at least three valid lags. With six positions, 240-angle fits use only
lags 1-3. Finite negative alpha values and low-R-squared fits remain included.

Motion labels use alpha below/above 1.2 to first define
diffusing/directional; diffusing tracks with speed rounded to two decimals
at or below 21.15 um/min become confined. Manual assignments classify track 39
as directional, and tracks 23 and 93 as confined. Labels stay fixed across
reconstruction rates. `analysis/motion_classification.csv`
records the rule for each identity.

Figure 4 f uses Gaussian KDE with 0.75 times Scott's bandwidth.
Its two-component Gaussian mixture fits the alpha samples, using
`GaussianMixture(n_components=2, n_init=20, random_state=0, reg_covar=1e-6)`.
The fitted means are approximately 1.04 and 1.41. Supplementary 10 b uses
common 0.05 alpha bins and the five-point kernel `(1,2,3,2,1)/9`.
Each panel exports one annotated PDF. Numerical curves, fit parameters, box
summaries and example selections are saved with the relevant panels.
