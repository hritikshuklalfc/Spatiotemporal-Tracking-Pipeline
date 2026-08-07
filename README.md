# Benchmarking Noise-Reduction Filters for Broadcast Football Tracking Data

This repository contains a data-processing and benchmarking pipeline for evaluating noise-reduction filters on broadcast football tracking data. It reconstructs player velocity from raw SkillCorner-style tracking data, injects synthetic corruption, applies several filtering methods, and measures both:

- kinematic velocity-recovery error, and
- downstream distortion of a Spearman pitch-control surface.

The central question is not simply which filter looks smoother, but which filter best preserves the tactical decision model inferred from the motion data.

## Key findings

The pipeline compares:

- Moving Average
- Savitzky-Golay
- 4th-order Butterworth
- Constant-velocity Kalman filter

In this noise regime, Moving Average performs best overall. It achieves strong velocity-recovery accuracy, produces lower pitch-control distortion, and does so at a lower computational cost than the alternatives.

## Repository structure

- Spearman_Pitch_Control/config.py: match paths, pitch geometry, filter settings, and pitch-control constants.
- Spearman_Pitch_Control/loader.py: loads tracking data, match metadata, and evaluation frames.
- Spearman_Pitch_Control/filters.py: builds the clean velocity baseline, injects synthetic noise, and runs the filters.
- Spearman_Pitch_Control/pitch_control.py: builds pitch-control surfaces, computes error maps, and generates evaluation tables.
- Spearman_Pitch_Control/visualiser.py: creates analytical figures and saves them to disk.
- Spearman_Pitch_Control/main.py: single-match entry point for the main paper tables and figures.
- Spearman_Pitch_Control/multi_match.py: cross-match runner for aggregated statistics and JSON outputs.
- helper_files/: utilities for tracking extraction and preprocessing.
- error_calculation/: standalone helpers for MAE, RMSE, and Pearson-r analysis.
- outputs/: generated tables, figures, and match-specific results.

## Pipeline overview

1. Load match data from the SkillCorner archive under data/archive_skillcorner/.
2. Reconstruct a clean velocity baseline from player tracking.
3. Inject synthetic Gaussian noise and 8 m displacement spikes to simulate broadcast-style corruption.
4. Apply the candidate filters.
5. Compare filtered velocities against the reconstructed ground truth.
6. Convert the motion into Spearman pitch-control surfaces.
7. Measure the spatial distortion between filtered and ground-truth pitch-control maps.
8. Export evaluation tables and figures.

## Quick start

From the project root, activate your virtual environment and run the analysis:

```bash
source .venv/bin/activate
python3 Spearman_Pitch_Control/main.py
```

To run the broader cross-match aggregation:

```bash
source .venv/bin/activate
python3 Spearman_Pitch_Control/multi_match.py
```

## Outputs

The scripts write generated tables and figures to the outputs/ directory. For example, the single-match pipeline produces:

- Table I: velocity recovery RMSE
- Table II: pitch-control distortion at the default beta value
- Table III: beta sensitivity summary
- Figure 1: raw vs filtered vs ground-truth velocity
- Figure 2: ground-truth pitch-control surface
- Figure 3: pitch-control difference map
- Figure 4: accuracy vs compute trade-off scatter

The multi-match runner additionally produces aggregate cross-match summaries and JSON outputs.

## Data expectations

The repository expects match data in a structure similar to:

- data/archive_skillcorner/<match_id>/
- outputs/match_<match_id>/

If you are using a different dataset layout, update the paths in Spearman_Pitch_Control/config.py accordingly.
