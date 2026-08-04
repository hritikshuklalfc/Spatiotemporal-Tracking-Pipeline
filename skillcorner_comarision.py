import json
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.stats import pearsonr

# locations/directory
match_ids = [
    "1886347", "1899585", "1925299", "1953632", "1996435",
    "2006229", "2011166", "2013725", "2015213", "2017461",
]

base_dir     = "data/archive_skillcorner"
filter_names = ['skillcorner', 'moving_avg', 'savitzky_golay',
                'butterworth', 'kalman']
lables = {
    'skillcorner'    : 'SkillCorner (baseline)',
    'moving_avg'     : 'Moving Average',
    'savitzky_golay' : 'Savitzky-Golay',
    'butterworth'    : 'Butterworth (2Hz)',
    'kalman'         : 'Kalman (CV)',
}

# filter parameters
dt = 0.1
ma_k = 3        
sg_window = 7
sg_poly = 3
bw_fc = 2.0
bw_order = 4
kf_q_std = 0.1
kf_r_std = 0.4
gt_sg_window = 15    
gt_sg_poly = 2
max_gap_fromes = 50   
min_frames = 10  

# loading json file
def load_raw_tracking(jsonl_path):
    players = {}
    
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            frame = d["frame"]
            for p in d["player_data"]:
                pid = str(p["player_id"])
                if pid not in players:
                    players[pid] = {"x":[], "y":[], "is_detected":[], "frame": []}
                players[pid]["x"].append(p["x"])
                players[pid]["y"].append(p["y"])
                players[pid]["is_detected"].append(bool(p["is_detected"]))
                players[pid]["frame"].append(frame)
                
    return players

# ground truth via lineat interpolation
def build_linear_interpolation_truth(x_raw, y_raw, is_detected):
    x_raw = np.array(x_raw, dtype=float)
    y_raw = np.array(y_raw, dtype=float)
    is_det = np.array(is_detected, dtype=bool)
    n = len(x_raw)
    
    x_truth = np.full(n, np.nan)                    
    y_truth = np.full(n, np.nan)
    
    x_truth[is_det] = x_raw[is_det]
    y_truth[is_det] = y_raw[is_det]
    
    det_idx = np.where(is_det)[0]
    if len(det_idx) < 2:
        return x_truth, y_truth
    
    for i in range(len(det_idx)-1):
        t0 = det_idx[i]
        t1 = det_idx[i + 1]
        gap = t1 - t0
        
        if gap <= 1:
            continue
        if gap > max_gap_fromes:
            continue
        
        for t in range(t0 + 1, t1):
            alpha = (t - t0) / (t1 - t0)
            x_truth[t] = x_truth[t0] + alpha * (x_truth[t1] - x_truth[t0])
            y_truth[t] = y_truth[t0] + alpha * (y_truth[t1] - y_truth[t0])
            
    return x_truth, y_truth

# apply 4 filter to full raw series
def apply_all_filter(x_raw, y_raw):
    x = np.array(x_raw, dtype=float)
    y = np.array(y_raw, dtype=float)
    
    x = _fill_nan(x)
    y = _fill_nan(y)
    
    results = {}
    
    # moving average
    k = ma_k
    kernel = np.ones(2*k+1)/(2*k+1)
    results['moving_avg'] = (
        np.convolve(x, kernel, mode = "same"),
        np.convolve(y, kernel, mode="same")
    )
    
    # savitzky golay
    if len(x)>sg_window:
        results["savitzky_golay"] = (
            savgol_filter(x, sg_window, sg_poly),
            savgol_filter(y, sg_window, sg_poly)
        )
    else:
        results["savitzky_golay"] = (x.copy(), y.copy())
        
    # butterworth
    nyq = (1/dt) / 2
    b, a = butter(bw_order, bw_fc/nyq, btype='low')
    
    if len(x) > 3 * max(len(a), len(b)):
        results["butterworth"] = filtfilt(b, a, x), filtfilt(b, a, y)
    else:
        results["butterworth"] = (x.copy(), y.copy())
        
    # kalman (steady_state)
    results["kalman"] = _kalman_ss(x, y)
    
    return results

def _fill_nan(arr):
    arr = arr.copy()
    nans = np.isnan(arr)
    if not nans.any():
        return arr
    idx = np.arange(len(arr))
    arr[nans] = np.interp(idx[nans], idx[~nans], arr[~nans])
    return arr

def _kalman_ss(x_obs, y_obs):
    f = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=float)
    
    h = np.array([
        [1, 0, 0, 0], 
        [0, 1, 0, 0]
    ], dtype=float)
    
    q = np.eye(4) * kf_q_std**2
    r = np.eye(2) * kf_r_std**2
    
    p = np.eye(4)
    
    for _ in range(50):
        p = f @ p @ f.T + q
        s = h @ p @ h.T + r 
        k = p @ h.T @ np.linalg.inv(s)
        p = (np.eye(4) - k @ h) @ p
        
    cx, cy, cvx, cvy = x_obs[0], y_obs[0], 0.0, 0.0
    xs, ys = [], []
    
    for i in range(len(x_obs)):
        px  = cx + cvx*dt;  py  = cy + cvy*dt
        pvx = cvx;           pvy = cvy
        rx  = x_obs[i] - px; ry = y_obs[i] - py
        cx  = px  + k[0,0]*rx + k[0,1]*ry
        cy  = py  + k[1,0]*rx + k[1,1]*ry
        cvx = pvx + k[2,0]*rx + k[2,1]*ry
        cvy = pvy + k[3,0]*rx + k[3,1]*ry
        xs.append(cx); ys.append(cy)
    
    return np.array(xs), np.array(ys)

# score at extrapolated frames only
def score_at_extrapolated(x_method, y_method, x_truth, y_truth, is_detected, sc_valid_mask=None):
    is_det = np.array(is_detected, dtype=bool)
    extrap = ~is_det
    
    vx_m = np.diff(x_method) / dt
    vy_m = np.diff(y_method) / dt
    vx_t = np.diff(x_truth) / dt
    vy_t = np.diff(y_truth) / dt
    
    speed_m = np.sqrt((vx_m**2) + (vy_m**2))
    speed_t = np.sqrt(vx_t**2 + vy_t**2)
    
    mask = extrap[1:]
    
    valid = mask & np.isfinite(speed_t)
    
    if sc_valid_mask is not None:
        sc_valid_vel = sc_valid_mask[1:]  
        valid = valid & sc_valid_vel
        
    if valid.sum() < min_frames:
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
    
# pre match runner
def run_match(match_id):
    jsonl_path = f"{base_dir}/{match_id}/{match_id}_tracking_extrapolated.jsonl"
    
    if not Path(jsonl_path).exists():
        print(f"Skip {match_id} -- No JSONL")
        return None
    
    print(f"\nMatch {match_id}....")
    players = load_raw_tracking(jsonl_path)
    
    accum = {f: {'MAE': [], 'RMSE': [], 'Pearson_r': [], 'n_frames':[]} for f in filter_names}
    
    n_scored = 0
    for pid, pdata in players.items():
        x_raw = pdata["x"]
        y_raw = pdata["y"]
        is_det = pdata["is_detected"]
        
        if len(x_raw) < 20:
            continue
        
        x_truth, y_truth = build_linear_interpolation_truth(x_raw, y_raw, is_det)
        
        x_raw_f = np.array(x_raw, dtype=float)
        y_raw_f = np.array(y_raw, dtype=float)
        
        # Calculate SkillCorner baseline scores
        s_sc = score_at_extrapolated(
            x_raw_f,
            y_raw_f,
            x_truth, y_truth, is_det
        )

        if s_sc is None:
            continue
        
        # Create mask based on where SkillCorner actually output valid coordinates
        sc_valid_mask = np.isfinite(x_raw_f) & np.isfinite(y_raw_f)
        
        for metric in ['MAE', 'RMSE', 'Pearson_r']:
            accum['skillcorner'][metric].append(s_sc[metric])
        accum['skillcorner']['n_frames'].append(s_sc['n_frames'])
        
        filtered = apply_all_filter(x_raw, y_raw)
        
        for fname, (xf, yf) in filtered.items():
            # Pass the SkillCorner valid mask to restrict evaluation frames
            s = score_at_extrapolated(xf, yf, x_truth, y_truth, is_det, sc_valid_mask=sc_valid_mask)
            if s is None:
                continue
            for metric in ['MAE', 'RMSE', 'Pearson_r']:
                accum[fname][metric].append(s[metric])
            accum[fname]['n_frames'].append(s["n_frames"])
                
        # Move outside the filter loop so we only increment once per player
        n_scored += 1
            
    print(f" Scored {n_scored}/{len(players)} players")
    
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
    
# aggrigate and table printing
def print_table_eight(all_results):
    valid = [r for r in all_results if r is not None]
    n = len(valid)
    
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
    
    for fname in filter_names:
        maes  = [r['metrics'][fname]['MAE']       for r in valid
                 if r['metrics'][fname]['MAE'] is not None]
        rmses = [r['metrics'][fname]['RMSE']      for r in valid
                 if r['metrics'][fname]['RMSE'] is not None]
        rs    = [r['metrics'][fname]['Pearson_r'] for r in valid
                 if r['metrics'][fname]['Pearson_r'] is not None]
        ns    = [r['metrics'][fname]['n_frames']  for r in valid
                 if r['metrics'][fname]['n_frames'] is not None]

        if not maes:
            print(f"{lables[fname]:<26} {'N/A':>10}")
            continue

        separator = "─" * 80 if fname == 'skillcorner' else ""
        if separator:
            print(separator)

        print(f"{lables[fname]:<26} "
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

    for fname in [f for f in filter_names if f != 'skillcorner']:
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

        print(f"{lables[fname]:<26} {mae_ratio:>10.3f} {r_ratio:>10.3f} "
              f"{interp:>20}")

    print("=" * 65)
    print("  MAE ratio < 1.0 → filter outperforms SkillCorner baseline")
    print("  MAE ratio > 1.0 → SkillCorner outperforms the filter")
    print("  r ratio   < 1.0 → filter has worse temporal shape fidelity")


def main():
    all_results = []

    for match_id in match_ids:
        result = run_match(match_id)
        all_results.append(result)

    print_table_eight(all_results)

    n_ok = sum(1 for r in all_results if r is not None)
    
    # FIXED: Was using len(match_id) instead of len(match_ids)
    print(f"\nCompleted: {n_ok}/{len(match_ids)} matches")

    for i, (mid, result) in enumerate(zip(match_ids, all_results)):
        status = "OK" if result is not None else "FAILED"
        # FIXED: Indentation applied to output properly inside the loop
        print(f"  {mid}: {status}")
    
if __name__ == "__main__":
    main()