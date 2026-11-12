"""
Noise Sensitivity Sweep
=======================
Tests filter rankings across a grid of noise parameters:
  sigma  ∈ {0.15, 0.25, 0.40, 0.60} m   (Gaussian positional noise)
  p_occ  ∈ {0.02, 0.05, 0.10}            (spike injection probability)

For each (sigma, p_occ) combination, runs all 10 matches and reports:
  - Mean RMSE per filter (global kinematic)
  - Mean MAE per filter  (global kinematic)  
  - RMSE rank of each filter
  - Whether MA rank-1 is stable

Output: console table + CSV saved to outputs/sensitivity_results.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, butter, filtfilt
from itertools import product

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DATA = PROJECT_ROOT / "data" / "archive_skillcorner"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "sensitivity_results.csv"
MATCH_IDS = [
    "1886347", "1899585", "1925299", "1953632", "1996435",
    "2006229", "2011166", "2013725", "2015213", "2017461",
]

# ── Fixed parameters (match paper) ────────────────────────────────────────
FPS        = 10
DT         = 1.0 / FPS
SPIKE_MAG  = 8.0
NOISE_SEED = 42

# MA: half-window=7 → full window=15
MA_HALF    = 7
# SG
SG_WINDOW  = 15
SG_POLY    = 2
# GT SG (ground truth construction)
GT_SG_WIN  = 15
GT_SG_POLY = 2
# Butterworth
BW_FC      = 1.5
BW_ORDER   = 2
# Kalman
KF_Q_STD   = 0.01
KF_R_STD   = 1.5

# ── Sweep grid ────────────────────────────────────────────────────────────
SIGMA_VALUES = [0.15, 0.25, 0.40, 0.60]
POCC_VALUES  = [0.02, 0.05, 0.10]

FILTER_NAMES  = ['moving_avg', 'savitzky_golay', 'butterworth', 'kalman']
FILTER_LABELS = {
    'moving_avg':     'Moving Average',
    'savitzky_golay': 'Savitzky-Golay',
    'butterworth':    'Butterworth',
    'kalman':         'Kalman (CV)',
}

# ── Filter implementations (self-contained, no config.py dependency) ──────

def build_ground_truth(x_clean, y_clean):
    x = np.array(x_clean, dtype=float)
    y = np.array(y_clean, dtype=float)
    xs = savgol_filter(x, window_length=GT_SG_WIN, polyorder=GT_SG_POLY)
    ys = savgol_filter(y, window_length=GT_SG_WIN, polyorder=GT_SG_POLY)
    vx = np.diff(xs) / DT
    vy = np.diff(ys) / DT
    return xs, ys, vx, vy

def inject_noise(x_star, y_star, sigma, p_occ, seed=NOISE_SEED):
    n   = len(x_star)
    rng = np.random.default_rng(seed)
    eta_x = rng.normal(0, sigma, size=n)
    eta_y = rng.normal(0, sigma, size=n)
    occ   = rng.uniform(0, 1, size=n) < p_occ
    d     = np.where(occ, SPIKE_MAG, 0.0)
    return x_star + eta_x + d, y_star + eta_y + d

def _vel(xf, yf):
    return np.diff(xf) / DT, np.diff(yf) / DT

def apply_moving_avg(xn, yn, half_window=MA_HALF):
    k      = 2 * half_window + 1
    kernel = np.ones(k) / k
    return _vel(np.convolve(xn, kernel, 'same'),
                np.convolve(yn, kernel, 'same'))

def apply_savgol(xn, yn, window=SG_WINDOW):
    return _vel(savgol_filter(xn, window, SG_POLY),
                savgol_filter(yn, window, SG_POLY))

def apply_butterworth(xn, yn, cutoff=BW_FC):
    nyq  = FPS / 2.0
    b, a = butter(BW_ORDER, cutoff / nyq, btype='low')
    return _vel(filtfilt(b, a, xn), filtfilt(b, a, yn))

def apply_kalman(xn, yn):
    F = np.array([[1,0,DT,0],[0,1,0,DT],[0,0,1,0],[0,0,0,1]], dtype=float)
    H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
    Q = np.eye(4) * KF_Q_STD**2
    R = np.eye(2) * KF_R_STD**2
    P = np.eye(4)
    for _ in range(50):
        P = F @ P @ F.T + Q
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        P = (np.eye(4) - K @ H) @ P
    cx, cy, cvx, cvy = xn[0], yn[0], 0.0, 0.0
    xs, ys = [], []
    for i in range(len(xn)):
        px, py   = cx + cvx*DT, cy + cvy*DT
        rx, ry   = xn[i] - px,   yn[i] - py
        cx  = px  + K[0,0]*rx + K[0,1]*ry
        cy  = py  + K[1,0]*rx + K[1,1]*ry
        cvx = cvx + K[2,0]*rx + K[2,1]*ry
        cvy = cvy + K[3,0]*rx + K[3,1]*ry
        xs.append(cx); ys.append(cy)
    return _vel(np.array(xs), np.array(ys))

FILTER_FNS = {
    'moving_avg':     apply_moving_avg,
    'savitzky_golay': apply_savgol,
    'butterworth':    apply_butterworth,
    'kalman':         apply_kalman,
}

# ── Per-player error computation ──────────────────────────────────────────

def player_errors(x_clean, y_clean, sigma, p_occ, filter_overrides=None):
    """Returns dict of {filter_name: (rmse, mae)} for one player."""
    x  = np.array(x_clean, dtype=float)
    y  = np.array(y_clean, dtype=float)
    filter_overrides = filter_overrides or {}
    sg_window = filter_overrides.get('savitzky_golay', {}).get('window', SG_WINDOW)
    if len(x) < max(SG_WINDOW, sg_window) + 2:
        return None

    xs, ys, vx_gt, vy_gt = build_ground_truth(x, y)
    xn, yn               = inject_noise(xs, ys, sigma, p_occ)
    v_gt = np.sqrt(vx_gt**2 + vy_gt**2)

    out = {}
    for fname, fn in FILTER_FNS.items():
        vxf, vyf = fn(xn, yn, **filter_overrides.get(fname, {}))
        vf       = np.sqrt(vxf**2 + vyf**2)
        n        = min(len(v_gt), len(vf))
        if n == 0:
            continue
        diff = v_gt[:n] - vf[:n]
        out[fname] = (
            float(np.sqrt(np.mean(diff**2))),   # RMSE
            float(np.mean(np.abs(diff)))         # MAE
        )
    return out

# ── Per-match aggregation ─────────────────────────────────────────────────

def run_match(match_id, sigma, p_occ):
    tracking_path = BASE_DATA / match_id / "extracted_player_tracking.json"
    with open(tracking_path) as f:
        tracking = json.load(f)

    rmse_acc = {fn: [] for fn in FILTER_NAMES}
    mae_acc  = {fn: [] for fn in FILTER_NAMES}

    for pid, pdata in tracking.items():
        result = player_errors(pdata['x'], pdata['y'], sigma, p_occ)
        if result is None:
            continue
        for fname, (rmse, mae) in result.items():
            rmse_acc[fname].append(rmse)
            mae_acc[fname].append(mae)

    means_rmse = {fn: np.mean(v) if v else np.nan for fn, v in rmse_acc.items()}
    means_mae  = {fn: np.mean(v) if v else np.nan for fn, v in mae_acc.items()}
    return means_rmse, means_mae

# ── Main sweep ────────────────────────────────────────────────────────────

def run_sweep():
    records = []

    total = len(SIGMA_VALUES) * len(POCC_VALUES)
    done  = 0

    for sigma, p_occ in product(SIGMA_VALUES, POCC_VALUES):
        done += 1
        print(f"\n[{done}/{total}] σ={sigma:.2f}m  p_occ={p_occ:.2f}", flush=True)

        all_rmse = {fn: [] for fn in FILTER_NAMES}
        all_mae  = {fn: [] for fn in FILTER_NAMES}

        for mid in MATCH_IDS:
            mr, mm = run_match(mid, sigma, p_occ)
            for fn in FILTER_NAMES:
                if not np.isnan(mr[fn]):
                    all_rmse[fn].append(mr[fn])
                if not np.isnan(mm[fn]):
                    all_mae[fn].append(mm[fn])
            print(f"  match {mid} done", flush=True)

        # Aggregate across 10 matches
        agg_rmse = {fn: np.mean(v) for fn, v in all_rmse.items()}
        agg_mae  = {fn: np.mean(v) for fn, v in all_mae.items()}

        # Ranks (1=best)
        rmse_rank = {fn: sorted(agg_rmse, key=agg_rmse.get).index(fn)+1
                     for fn in FILTER_NAMES}
        mae_rank  = {fn: sorted(agg_mae,  key=agg_mae.get).index(fn)+1
                     for fn in FILTER_NAMES}

        row = {'sigma': sigma, 'p_occ': p_occ}
        for fn in FILTER_NAMES:
            row[f'rmse_{fn}']      = round(agg_rmse[fn], 4)
            row[f'mae_{fn}']       = round(agg_mae[fn],  4)
            row[f'rmse_rank_{fn}'] = rmse_rank[fn]
            row[f'mae_rank_{fn}']  = mae_rank[fn]
        records.append(row)

    return pd.DataFrame(records)

# ── Pretty-print results ──────────────────────────────────────────────────

def print_summary(df):
    print("\n" + "="*80)
    print("NOISE SENSITIVITY SWEEP — RMSE Rankings (1=best)")
    print("="*80)
    print(f"{'σ':>5}  {'p_occ':>6}  {'MA':>4}  {'SG':>4}  {'BW':>4}  {'KF':>4}  MA_wins_RMSE?  MA_wins_MAE?")
    print("-"*80)
    for _, r in df.iterrows():
        ma_rmse = r['rmse_rank_moving_avg']
        ma_mae  = r['mae_rank_moving_avg']
        print(f"{r['sigma']:>5.2f}  {r['p_occ']:>6.2f}  "
              f"{int(r['rmse_rank_moving_avg']):>4}  "
              f"{int(r['rmse_rank_savitzky_golay']):>4}  "
              f"{int(r['rmse_rank_butterworth']):>4}  "
              f"{int(r['rmse_rank_kalman']):>4}  "
              f"{'YES' if ma_rmse==1 else 'NO':>13}  "
              f"{'YES' if ma_mae==1  else 'NO':>12}")
    print("="*80)

    print("\n" + "="*80)
    print("RMSE VALUES PER CONDITION (m/s)")
    print("="*80)
    print(f"{'σ':>5}  {'p_occ':>6}  {'MA':>8}  {'SG':>8}  {'BW':>8}  {'KF':>8}")
    print("-"*80)
    for _, r in df.iterrows():
        print(f"{r['sigma']:>5.2f}  {r['p_occ']:>6.2f}  "
              f"{r['rmse_moving_avg']:>8.4f}  "
              f"{r['rmse_savitzky_golay']:>8.4f}  "
              f"{r['rmse_butterworth']:>8.4f}  "
              f"{r['rmse_kalman']:>8.4f}")
    print("="*80)

    # Summary: how often does MA rank 1?
    ma_rmse_wins = (df['rmse_rank_moving_avg'] == 1).sum()
    ma_mae_wins  = (df['mae_rank_moving_avg']  == 1).sum()
    total        = len(df)
    print(f"\nMA rank-1 RMSE: {ma_rmse_wins}/{total} conditions")
    print(f"MA rank-1 MAE:  {ma_mae_wins}/{total} conditions")

    print("\n" + "="*80)
    print("MAE VALUES PER CONDITION (m/s)")
    print("="*80)
    print(f"{'σ':>5}  {'p_occ':>6}  {'MA':>8}  {'SG':>8}  {'BW':>8}  {'KF':>8}")
    print("-"*80)
    for _, r in df.iterrows():
        print(f"{r['sigma']:>5.2f}  {r['p_occ']:>6.2f}  "
              f"{r['mae_moving_avg']:>8.4f}  "
              f"{r['mae_savitzky_golay']:>8.4f}  "
              f"{r['mae_butterworth']:>8.4f}  "
              f"{r['mae_kalman']:>8.4f}")
    print("="*80)


def run_parameter_sweep(parameter, values):
    """Evaluate one filter parameter across all players and ten matches."""
    filter_name = 'moving_avg' if parameter == 'k' else (
        'savitzky_golay' if parameter == 'sg_window' else 'butterworth')
    records = []

    for value in values:
        rmse_values = []
        mae_values = []
        for mid in MATCH_IDS:
            tracking_path = BASE_DATA / mid / "extracted_player_tracking.json"
            with open(tracking_path) as f:
                tracking = json.load(f)
            overrides = {
                filter_name: ({'half_window': (value - 1) // 2} if parameter == 'k' else
                              {'window': value} if parameter == 'sg_window' else
                              {'cutoff': value})
            }
            for pdata in tracking.values():
                result = player_errors(pdata['x'], pdata['y'], 0.40, 0.05,
                                       overrides)
                if result and filter_name in result:
                    rmse_values.append(result[filter_name][0])
                    mae_values.append(result[filter_name][1])
        record = {
            'value': value,
            'rmse': float(np.mean(rmse_values)) if rmse_values else np.nan,
            'mae': float(np.mean(mae_values)) if mae_values else np.nan,
        }
        records.append(record)
        print(f"  {parameter}={value:<5} RMSE={record['rmse']:.6f} "
              f"MAE={record['mae']:.6f}", flush=True)
    return records


def print_parameter_sweeps():
    print("\n" + "="*80)
    print("HYPERPARAMETER SWEEP LOGS (mean across all players and 10 matches)")
    print("="*80)
    print("Moving Average k sweep")
    run_parameter_sweep('k', [5, 7, 9, 11, 13, 15, 21])
    print("\nSavitzky-Golay k sweep (window length)")
    run_parameter_sweep('sg_window', [5, 7, 9, 11, 13, 15, 21])
    print("\nButterworth fc sweep (Hz)")
    run_parameter_sweep('fc', [1.0, 1.5, 2.0, 3.0])
    print("="*80)

if __name__ == "__main__":
    print("Starting noise sensitivity sweep...")
    print(f"Data path: {BASE_DATA}")
    print(f"Output path: {OUTPUT_PATH}")
    print(f"Grid: {len(SIGMA_VALUES)} sigma values × {len(POCC_VALUES)} p_occ values × 10 matches")
    print(f"Total match runs: {len(SIGMA_VALUES)*len(POCC_VALUES)*len(MATCH_IDS)}\n")

    df = run_sweep()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print_summary(df)
    print_parameter_sweeps()
    print(f"\nResults saved to {OUTPUT_PATH}")