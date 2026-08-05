# Spearman Pitch Control

This project studies how tracking noise changes football pitch-control estimates. It reconstructs player velocity from raw SkillCorner-style data, applies several filters, and measures both the velocity recovery error and the downstream distortion in pitch-control surfaces.

The paper-level question is simple: which filter does the best job of preserving the football decision model, not just the smoothest-looking trajectory?

## Brief Paper Summary

The core result is that Kalman filtering gives the best end-to-end trade-off in this setup. It recovers velocity more accurately than the other tested filters and also produces the lowest pitch-control distortion. Simpler filters like moving average are faster, but they preserve the downstream model less well.

## Repository Architecture

The code is organised as a small research pipeline rather than a framework:

- [Spearman_Pitch_Control/config.py](Spearman_Pitch_Control/config.py): match paths, pitch geometry, filter settings, and pitch-control constants.
- [Spearman_Pitch_Control/loader.py](Spearman_Pitch_Control/loader.py): reads tracking, match metadata, and evaluation frames.
- [Spearman_Pitch_Control/filters.py](Spearman_Pitch_Control/filters.py): builds the clean velocity baseline, injects synthetic noise, and runs the filters.
- [Spearman_Pitch_Control/pitch_control.py](Spearman_Pitch_Control/pitch_control.py): builds pitch-control surfaces, computes MAE maps, and prints Table I.
- [Spearman_Pitch_Control/visualiser.py](Spearman_Pitch_Control/visualiser.py): saves the figures into the outputs folder.
- [Spearman_Pitch_Control/main.py](Spearman_Pitch_Control/main.py): single-match entrypoint used for the main paper tables and figures.
- [Spearman_Pitch_Control/multi_match.py](Spearman_Pitch_Control/multi_match.py): cross-match runner used for Tables IV to VI and the aggregate JSON output.
- [helper_files/](helper_files/): utilities for tracking extraction and preprocessing.
- [error_calculation/](error_calculation/): standalone MAE, RMSE, and Pearson-r helpers.
- [data/](data/): raw and extracted match data.
- [outputs/](outputs/): generated figures, tables, and cross-match results.

## What To Run

Run everything from the project root after activating the virtual environment.

```bash
source .venv/bin/activate
python Spearman_Pitch_Control/main.py
```

Use the main pipeline for the single-match paper results. It prints the primary tables and saves the figures.

To run the broader cross-match aggregation, use:

```bash
python Spearman_Pitch_Control/multi_match.py
```

## Output Map

| Runner                                                                         | What it produces                                                     |
| ------------------------------------------------------------------------------ | -------------------------------------------------------------------- |
| [Spearman_Pitch_Control/main.py](Spearman_Pitch_Control/main.py)               | Table I: velocity recovery RMSE                                      |
| [Spearman_Pitch_Control/main.py](Spearman_Pitch_Control/main.py)               | Table II: pitch-control distortion at the default beta value         |
| [Spearman_Pitch_Control/main.py](Spearman_Pitch_Control/main.py)               | Table III: beta sensitivity summary                                  |
| [Spearman_Pitch_Control/main.py](Spearman_Pitch_Control/main.py)               | Figure 1: raw vs filtered vs ground-truth velocity                   |
| [Spearman_Pitch_Control/main.py](Spearman_Pitch_Control/main.py)               | Figure 2: ground-truth pitch-control surface                         |
| [Spearman_Pitch_Control/main.py](Spearman_Pitch_Control/main.py)               | Figure 3: pitch-control difference map                               |
| [Spearman_Pitch_Control/main.py](Spearman_Pitch_Control/main.py)               | Figure 4: accuracy vs compute trade-off scatter                      |
| [Spearman_Pitch_Control/multi_match.py](Spearman_Pitch_Control/multi_match.py) | Table IV: cross-match velocity recovery summary                      |
| [Spearman_Pitch_Control/multi_match.py](Spearman_Pitch_Control/multi_match.py) | Table V: cross-match pitch-control distortion                        |
| [Spearman_Pitch_Control/multi_match.py](Spearman_Pitch_Control/multi_match.py) | Table VI: ranking consistency across matches                         |
| [Spearman_Pitch_Control/multi_match.py](Spearman_Pitch_Control/multi_match.py) | [outputs/cross_match_results.json](outputs/cross_match_results.json) |

The figures saved by the main runner are written to [outputs/](outputs/), with these default filenames:

- [outputs/fig1_velocity_comparison.png](outputs/fig1_velocity_comparison.png)
- [outputs/fig2_pitch_control.png](outputs/fig2_pitch_control.png)
- [outputs/fig3*diff_Kalman*%28CV%29_vs_Savitzky-Golay.png](outputs/fig3_diff_Kalman_%28CV%29_vs_Savitzky-Golay.png)
- [outputs/fig4_pareto_scatter.png](outputs/fig4_pareto_scatter.png)

## How The Pipeline Works

1. Load match data from [data/archive_skillcorner/](data/archive_skillcorner/).
2. Reconstruct a clean velocity baseline from player tracking.
3. Inject synthetic Gaussian noise and spikes to simulate corruption.
4. Apply moving average, Savitzky-Golay, Butterworth, and Kalman filtering.
5. Compare the filtered velocity against the reconstructed ground truth.
6. Convert filtered motion into pitch-control surfaces.
7. Measure the distortion between the filtered and ground-truth pitch-control maps.
8. Print the paper tables and save the figures to [outputs/](outputs/).

## Notes

The default match in the configuration is the one currently used for the single-match pipeline. The multi-match runner is the better entrypoint when you want aggregated statistics across the available matches.
