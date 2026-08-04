# Spearman Pitch Control: Filtering Noise for Pitch-Control Distortion Analysis

## Aim

This project investigates how different smoothing and filtering methods affect the recovery of player velocity from noisy tracking data and, more importantly, how those changes distort downstream pitch-control models. The central question is not simply which filter produces the smoothest trajectory, but which filter preserves the football decision model most faithfully.

## Objectives

- Reconstruct player velocity from raw tracking data.
- Inject controlled synthetic noise and spikes to create a realistic recovery task.
- Compare multiple filtering methods including moving average, Savitzky-Golay, Butterworth, and Kalman.
- Evaluate each method using velocity error metrics such as MAE, RMSE, and Pearson correlation.
- Measure how filtering choices alter the resulting pitch-control surface.
- Identify the best trade-off between reconstruction accuracy and computational cost.

## What We Did

The pipeline was implemented as a compact research workflow that follows these steps:

1. Load match data from the SkillCorner-style archive under the data folder.
2. Read player tracking, team mapping, and active evaluation frames.
3. Build a velocity baseline using smoothing-based reconstruction.
4. Inject synthetic Gaussian noise and occasional spikes to simulate tracking noise.
5. Apply several filters to each player track:
   - Moving Average
   - Savitzky-Golay
   - Butterworth
   - Kalman (Constant Velocity)
6. Compare filtered velocities against the reconstructed ground truth using error metrics.
7. Convert the resulting velocities into pitch-control surfaces and measure the downstream distortion.
8. Generate figures and summary tables for interpretation and reporting.

## Project Structure

- Spearman_Pitch_Control/: core pipeline for pitch-control analysis and filtering
- helper_files/: support utilities for extracting and processing tracking features
- error_calculation/: scripts for MAE, RMSE, and Pearson-based evaluation
- data/: match tracking and metadata files
- outputs/: generated figures and result summaries
- skillcorner_comarision.py: comparison script for evaluating filters against SkillCorner-style data

## Main Findings

In the validated experiments, Kalman filtering produced the best overall performance in terms of both velocity recovery and downstream pitch-control fidelity. Moving average was computationally cheaper but less accurate. Savitzky-Golay and Butterworth improved smoothing to some degree, but did not preserve the pitch-control model as well as Kalman in this setup.

## How to Run

From the project root, run:

```bash
python skillcorner_comarision.py
```

Or use the main analysis pipeline in the Spearman_Pitch_Control folder.

## Notes

The project was developed as a paper-oriented research workflow rather than a production-grade package. The code was also cleaned up to make multi-match runs more reliable, improve validation of active frames, and prevent crashes during evaluation.
