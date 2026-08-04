"""
option5_skillcorner_comparison.py — Section V.G

Compares SkillCorner's proprietary extrapolation model against
the four benchmark filters at extrapolated frames only.

Ground truth: linear interpolation between surrounding detected frames.
This is independent of both SkillCorner's model and our filters — fair.

Evaluated at: extrapolated frames only (is_detected == False).
Detected frames excluded — SkillCorner IS the camera there, not a model.

Produces Table VIII.
"""

import json
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.stats import pearsonr

MATCH_IDS = [
    "1886347", "1899585", "1925299", "1953632", "1996435",
    "2006229", "2011166", "2013725", "2015213", "2017461",
]
BASE_DIR     = "data/archive_skillcorner"
FILTER_NAMES = ['skillcorner', 'moving_avg', 'savitzky_golay',
                'butterworth', 'kalman']
LABELS = {
    'skillcorner'    : 'SkillCorner (baseline)',
    'moving_avg'     : 'Moving Average',
    'savitzky_golay' : 'Savitzky-Golay',
    'butterworth'    : 'Butterworth (2Hz)',
    'kalman'         : 'Kalman (CV)',
}

# Filter parameters — must match config.py
DT          = 0.1
MA_K        = 3        # window = 2k+1 = 7
SG_WINDOW   = 7
SG_POLY     = 3
BW_FC       = 2.0
BW_ORDER    = 4
KF_Q_STD    = 0.1
KF_R_STD    = 0.4

# Ground truth construction
GT_SG_WINDOW  = 15    # wider window for ground truth smoothing
GT_SG_POLY    = 2
MAX_GAP_FRAMES = 50   # exclude gaps longer than 5s (unreliable linear interp)
MIN_FRAMES     = 10   # minimum extrapolated frames to score a player


# ─────────────────────────────────────────────────────────────
# STEP 1 — Load raw JSONL and extract per-player arrays
# ─────────────────────────────────────────────────────────────

def load_raw_tracking(jsonl_path):
    """
    Reads tracking_extrapolated.jsonl.
    Returns {player_id_str -> {'x':[], 'y':[], 'is_detected':[], 'frame':[]}}
    Memory-efficient: reads line by line.
    """
    players = {}

    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            frame = d['frame']
            for p in d['player_data']:
                pid = str(p['player_id'])
                if pid not in players:
                    players[pid] = {'x': [], 'y': [],
                                    'is_detected': [], 'frame': []}
                players[pid]['x'].append(p['x'])
                players[pid]['y'].append(p['y'])
                players[pid]['is_detected'].append(bool(p['is_detected']))
                players[pid]['frame'].append(frame)

    return players


# ─────────────────────────────────────────────────────────────
# STEP 2 — Build ground truth via linear interpolation
# ─────────────────────────────────────────────────────────────

def build_linear_interpolation_truth(x_raw, y_raw, is_detected):
    """
    Constructs pseudo-ground-truth by linearly interpolating
    between surrounding detected frames.

    This is independent of SkillCorner's model and our filters.
    Rationale: assumes near-linear motion between detections,
    which is a neutral, model-free baseline.

    Returns x_truth, y_truth arrays (same length as input).
    NaN where gap exceeds MAX_GAP_FRAMES.
    """
    x_raw = np.array(x_raw, dtype=float)
    y_raw = np.array(y_raw, dtype=float)
    is_det = np.array(is_detected, dtype=bool)
    n = len(x_raw)

    x_truth = np.full(n, np.nan)
    y_truth = np.full(n, np.nan)

    # Copy detected frames directly
    x_truth[is_det] = x_raw[is_det]
    y_truth[is_det] = y_raw[is_det]

    # Find detected frame indices
    det_idx = np.where(is_det)[0]
    if len(det_idx) < 2:
        return x_truth, y_truth

    # Linear interpolation across each gap
    for i in range(len(det_idx) - 1):
        t0 = det_idx[i]
        t1 = det_idx[i + 1]
        gap = t1 - t0

        if gap <= 1:
            continue   # consecutive detected frames, no gap
        if gap > MAX_GAP_FRAMES:
            continue   # gap too long, linear interp unreliable

        # Interpolate each frame between t0 and t1
        for t in range(t0 + 1, t1):
            alpha = (t - t0) / (t1 - t0)
            x_truth[t] = x_truth[t0] + alpha * (x_truth[t1] - x_truth[t0])
            y_truth[t] = y_truth[t0] + alpha * (y_truth[t1] - y_truth[t0])

    return x_truth, y_truth


# ─────────────────────────────────────────────────────────────
# STEP 3 — Apply four filters to full raw series
# ─────────────────────────────────────────────────────────────

def apply_all_filters(x_raw, y_raw):
    """
    Applies four filters to the full raw coordinate series.
    No noise injection — raw SkillCorner positions are the input.
    Returns dict of filtered (x, y) arrays.
    """
    x = np.array(x_raw, dtype=float)
    y = np.array(y_raw, dtype=float)

    # Handle NaN in raw data before filtering
    x = _fill_nan(x)
    y = _fill_nan(y)

    results = {}

    # Moving Average
    k = MA_K
    kernel = np.ones(2*k+1) / (2*k+1)
    results['moving_avg'] = (
        np.convolve(x, kernel, mode='same'),
        np.convolve(y, kernel, mode='same')
    )

    # Savitzky-Golay
    if len(x) > SG_WINDOW:
        results['savitzky_golay'] = (
            savgol_filter(x, SG_WINDOW, SG_POLY),
            savgol_filter(y, SG_WINDOW, SG_POLY)
        )
    else:
        results['savitzky_golay'] = (x.copy(), y.copy())

    # Butterworth
    nyq  = (1/DT) / 2
    b, a = butter(BW_ORDER, BW_FC/nyq, btype='low')
    if len(x) > 3 * max(len(a), len(b)):
        results['butterworth'] = (filtfilt(b, a, x), filtfilt(b, a, y))
    else:
        results['butterworth'] = (x.copy(), y.copy())

    # Kalman (steady-state)
    results['kalman'] = _kalman_ss(x, y)

    return results


def _fill_nan(arr):
    """Linear interpolation of NaN values for filter stability."""
    arr = arr.copy()
    nans = np.isnan(arr)
    if not nans.any():
        return arr
    idx = np.arange(len(arr))
    arr[nans] = np.interp(idx[nans], idx[~nans], arr[~nans])
    return arr


def _kalman_ss(x_obs, y_obs):
    """Steady-state Kalman filter — same as filters.py implementation."""
    F = np.array([[1,0,DT,0],[0,1,0,DT],
                  [0,0,1,0],[0,0,0,1]], dtype=float)
    H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
    Q = np.eye(4) * KF_Q_STD**2
    R = np.eye(2) * KF_R_STD**2

    P = np.eye(4)
    for _ in range(50):
        P = F @ P @ F.T + Q
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        P = (np.eye(4) - K @ H) @ P

    cx, cy, cvx, cvy = x_obs[0], y_obs[0], 0.0, 0.0
    xs, ys = [], []

    for i in range(len(x_obs)):
        px  = cx + cvx*DT;  py  = cy + cvy*DT
        pvx = cvx;           pvy = cvy
        rx  = x_obs[i] - px; ry = y_obs[i] - py
        cx  = px  + K[0,0]*rx + K[0,1]*ry
        cy  = py  + K[1,0]*rx + K[1,1]*ry
        cvx = pvx + K[2,0]*rx + K[2,1]*ry
        cvy = pvy + K[3,0]*rx + K[3,1]*ry
        xs.append(cx); ys.append(cy)

    return np.array(xs), np.array(ys)


# ─────────────────────────────────────────────────────────────
# STEP 4 — Score at extrapolated frames only
# ─────────────────────────────────────────────────────────────

def score_at_extrapolated(x_method, y_method,
                           x_truth, y_truth,
                           is_detected):
    """
    Computes velocity from method positions and scores against
    truth velocity — evaluated at extrapolated frames only.

    Velocity at frame i = displacement from i-1 to i.
    Extrapolated mask shifted by 1 to align with velocity indexing.
    """
    is_det  = np.array(is_detected, dtype=bool)
    extrap  = ~is_det

    # Velocity arrays (length N-1)
    vx_m = np.diff(x_method) / DT
    vy_m = np.diff(y_method) / DT
    vx_t = np.diff(x_truth)  / DT
    vy_t = np.diff(y_truth)  / DT

    speed_m = np.sqrt(vx_m**2 + vy_m**2)
    speed_t = np.sqrt(vx_t**2 + vy_t**2)

    # Align mask to velocity indexing
    mask = extrap[1:]   # frame i velocity corresponds to extrap status of frame i

    # Remove NaN truth frames (gaps beyond MAX_GAP_FRAMES)
    valid = mask & np.isfinite(speed_t)

    if valid.sum() < MIN_FRAMES:
        return None

    sm = speed_m[valid]
    st = speed_t[valid]

    mae_val  = float(np.mean(np.abs(sm - st)))
    rmse_val = float(np.sqrt(np.mean((sm - st)**2)))
    r_val    = float(pearsonr(sm, st)[0]) \
               if np.std(sm) > 1e-9 and np.std(st) > 1e-9 else 0.0

    return {
        'MAE'      : mae_val,
        'RMSE'     : rmse_val,
        'Pearson_r': r_val,
        'n_frames' : int(valid.sum())
    }


# ─────────────────────────────────────────────────────────────
# STEP 5 — Per-match runner
# ─────────────────────────────────────────────────────────────

def run_match(match_id):
    jsonl_path = f"{BASE_DIR}/{match_id}/{match_id}_tracking_extrapolated.jsonl"

    if not Path(jsonl_path).exists():
        print(f"  [SKIP] {match_id} — no JSONL")
        return None

    print(f"\nMatch {match_id}...")
    players = load_raw_tracking(jsonl_path)

    accum = {f: {'MAE': [], 'RMSE': [], 'Pearson_r': [], 'n_frames': []}
             for f in FILTER_NAMES}

    n_scored = 0
    for pid, pdata in players.items():
        x_raw    = pdata['x']
        y_raw    = pdata['y']
        is_det   = pdata['is_detected']

        if len(x_raw) < 20:
            continue

        # Build ground truth
        x_truth, y_truth = build_linear_interpolation_truth(
            x_raw, y_raw, is_det)

        # SkillCorner baseline — their own extrapolated positions
        s_sc = score_at_extrapolated(
            np.array(x_raw, dtype=float),
            np.array(y_raw, dtype=float),
            x_truth, y_truth, is_det)

        if s_sc is None:
            continue

        for metric in ['MAE', 'RMSE', 'Pearson_r']:
            accum['skillcorner'][metric].append(s_sc[metric])
        accum['skillcorner']['n_frames'].append(s_sc['n_frames'])

        # Four filters — applied to raw positions
        filtered = apply_all_filters(x_raw, y_raw)

        for fname, (xf, yf) in filtered.items():
            s = score_at_extrapolated(xf, yf, x_truth, y_truth, is_det)
            if s is None:
                continue
            for metric in ['MAE', 'RMSE', 'Pearson_r']:
                accum[fname][metric].append(s[metric])
            accum[fname]['n_frames'].append(s['n_frames'])

        n_scored += 1

    print(f"  Scored {n_scored}/{len(players)} players")

    return {
        'match_id' : match_id,
        'n_players': n_scored,
        'metrics'  : {
            fname: {
                metric: float(np.mean(vals)) if vals else None
                for metric, vals in m.items()
                if metric != 'n_frames'
            } | {'n_frames': int(sum(m['n_frames'])) if m['n_frames'] else 0}
            for fname, m in accum.items()
        }
    }


# ─────────────────────────────────────────────────────────────
# STEP 6 — Aggregate and print Table VIII
# ─────────────────────────────────────────────────────────────

def print_table_eight(all_results):
    valid = [r for r in all_results if r is not None]
    n     = len(valid)

    print(f"\n{'='*80}")
    print(f"TABLE VIII — Velocity Recovery at Extrapolated Frames: "
          f"Filters vs SkillCorner (N={n} matches)")
    print(f"{'='*80}")
    print(f"  Ground truth: linear interpolation between detected frames")
    print(f"  Evaluated at: extrapolated frames only (is_detected == False)")
    print(f"{'='*80}")
    print(f"{'Method':<26} {'MAE (m/s)':>10} {'RMSE (m/s)':>11} "
          f"{'Pearson r':>10} {'Total N':>9}")
    print("-" * 80)

    for fname in FILTER_NAMES:
        maes  = [r['metrics'][fname]['MAE']       for r in valid
                 if r['metrics'][fname]['MAE'] is not None]
        rmses = [r['metrics'][fname]['RMSE']      for r in valid
                 if r['metrics'][fname]['RMSE'] is not None]
        rs    = [r['metrics'][fname]['Pearson_r'] for r in valid
                 if r['metrics'][fname]['Pearson_r'] is not None]
        ns    = [r['metrics'][fname]['n_frames']  for r in valid
                 if r['metrics'][fname]['n_frames'] is not None]

        if not maes:
            print(f"{LABELS[fname]:<26} {'N/A':>10}")
            continue

        separator = "─" * 80 if fname == 'skillcorner' else ""
        if separator:
            print(separator)

        print(f"{LABELS[fname]:<26} "
              f"{np.mean(maes):>10.4f} "
              f"{np.mean(rmses):>11.4f} "
              f"{np.mean(rs):>10.4f} "
              f"{sum(ns):>9,}")

        if fname == 'skillcorner':
            print("─" * 80)

    print("=" * 80)

    # Key ratio table — filters vs SkillCorner
    sc_mae = np.mean([r['metrics']['skillcorner']['MAE'] for r in valid
                      if r['metrics']['skillcorner']['MAE'] is not None])
    sc_r   = np.mean([r['metrics']['skillcorner']['Pearson_r'] for r in valid
                      if r['metrics']['skillcorner']['Pearson_r'] is not None])

    print(f"\n{'='*65}")
    print("KEY RATIO — Filter vs SkillCorner baseline")
    print(f"{'='*65}")
    print(f"{'Method':<26} {'MAE ratio':>10} {'r ratio':>10} "
          f"{'MAE interpretation':>20}")
    print("-" * 65)

    for fname in [f for f in FILTER_NAMES if f != 'skillcorner']:
        maes = [r['metrics'][fname]['MAE'] for r in valid
                if r['metrics'][fname]['MAE'] is not None]
        rs   = [r['metrics'][fname]['Pearson_r'] for r in valid
                if r['metrics'][fname]['Pearson_r'] is not None]

        if not maes:
            continue

        mae_ratio = np.mean(maes) / sc_mae
        r_ratio   = np.mean(rs)   / sc_r

        if mae_ratio < 0.95:
            interp = "Filter BEATS SkillCorner"
        elif mae_ratio > 1.05:
            interp = "SkillCorner wins"
        else:
            interp = "Statistically tied"

        print(f"{LABELS[fname]:<26} {mae_ratio:>10.3f} {r_ratio:>10.3f} "
              f"{interp:>20}")

    print("=" * 65)
    print("  MAE ratio < 1.0 → filter outperforms SkillCorner baseline")
    print("  MAE ratio > 1.0 → SkillCorner outperforms the filter")
    print("  r ratio   < 1.0 → filter has worse temporal shape fidelity")


def main():
    all_results = []

    for match_id in MATCH_IDS:
        result = run_match(match_id)
        all_results.append(result)

    print_table_eight(all_results)

    n_ok = sum(1 for r in all_results if r is not None)
    print(f"\nCompleted: {n_ok}/{len(MATCH_IDS)} matches")


if __name__ == "__main__":
    main()