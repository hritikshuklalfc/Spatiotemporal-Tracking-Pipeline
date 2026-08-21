import time
import numpy as np
from scipy.stats import pearsonr

def mae(v_hat, v_star):
    return float(np.mean(np.abs(v_hat - v_star)))


def rmse(v_hat, v_star):
    return float(np.sqrt(np.mean((v_hat - v_star) ** 2)))


def pearson_r(v_hat, v_star):
    if np.std(v_hat) < 1e-9 or np.std(v_star) < 1e-9:
        return 0.0  # flat signal — correlation undefined
    r, _ = pearsonr(v_hat, v_star)
    return float(r)


def _to_speed(v):
    return np.sqrt(v[0]**2 + v[1]**2)

def score_filter(v_hat, v_star, mask=None):
    speed_hat  = _to_speed(v_hat)
    speed_star = _to_speed(v_star)

    n = min(len(speed_hat), len(speed_star))
    if n < 10:
        return {'MAE': np.nan, 'RMSE': np.nan, 'Pearson_r': np.nan}

    speed_hat  = speed_hat[:n]
    speed_star = speed_star[:n]

    # If no mask provided, use all frames
    if mask is not None:
        mask = np.array(mask[:n], dtype=bool)
        speed_hat  = speed_hat[mask]
        speed_star = speed_star[mask]

    if len(speed_hat) < 10:
        return {'MAE': np.nan, 'RMSE': np.nan, 'Pearson_r': np.nan}

    return {
        'MAE'      : mae(speed_hat,       speed_star),
        'RMSE'     : rmse(speed_hat,      speed_star),
        'Pearson_r': pearson_r(speed_hat, speed_star),
    }
    
def scored_filter_masked(v_hat, v_star, mask=None):
    speed_hat  = _to_speed(v_hat)
    speed_star = _to_speed(v_star)

    n = min(len(speed_hat), len(speed_star), len(mask))
    speed_hat  = speed_hat[:n]
    speed_star = speed_star[:n]
    mask       = mask[:n]

    speed_hat_m  = speed_hat[mask]
    speed_star_m = speed_star[mask]

    if len(speed_hat_m) < 10:
        return None

    return {
        'MAE'      : mae(speed_hat_m,       speed_star_m),
        'RMSE'     : rmse(speed_hat_m,      speed_star_m),
        'Pearson_r': pearson_r(speed_hat_m, speed_star_m),
        'n_frames' : int(mask.sum())
    }

def time_filter(filter_fn, x_noisy, y_noisy, n_runs=5):
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        filter_fn(x_noisy, y_noisy)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return float(np.mean(times))


def build_table_one(vel_cache, timing_results):
    filter_names = ['moving_avg', 'savitzky_golay',
                    'butterworth', 'kalman']
    labels = {
        'moving_avg'     : 'Moving Average',
        'savitzky_golay' : 'Savitzky-Golay',
        'butterworth'    : 'Butterworth (1.5Hz)',
        'kalman'         : 'Kalman (CV)',
    }

    accum = {f: {'MAE': [], 'RMSE': [], 'Pearson_r': []}
             for f in filter_names}

    for pid, result in vel_cache.items():
        if result is None:
            continue
        v_star = result['ground_truth']

        for fname in filter_names:
            v_hat = result[fname]
            s = score_filter(v_hat, v_star)
            for metric, val in s.items():
                if np.isfinite(val):
                    accum[fname][metric].append(val)

    print("\n" + "=" * 70)
    print("TABLE I — Velocity Recovery Error and Compute Cost")
    print("=" * 70)
    print(f"{'Filter':<22} {'MAE':>8} {'RMSE':>8} "
          f"{'Pearson r':>10} {'Time (s)':>10}")
    print("-" * 70)

    for fname in filter_names:
        a = accum[fname]
        mean_mae = np.mean(a['MAE'])      if a['MAE']       else float('nan')
        mean_rms = np.mean(a['RMSE'])     if a['RMSE']      else float('nan')
        mean_r   = np.mean(a['Pearson_r'])if a['Pearson_r'] else float('nan')
        t        = timing_results.get(fname, float('nan'))
        label    = labels[fname]
        print(f"{label:<22} {mean_mae:>8.4f} {mean_rms:>8.4f} "
              f"{mean_r:>10.4f} {t:>10.4f}")

    print("=" * 70)
    return accum

def build_table_seven(detected_accum, extrap_accum):
    filter_names = ['moving_avg', 'savitzky_golay', 'butterworth', 'kalman']
    labels = {
        'moving_avg'     : 'Moving Average',
        'savitzky_golay' : 'Savitzky-Golay',
        'butterworth'    : 'Butterworth (1.5Hz)',
        'kalman'         : 'Kalman (CV)',
    }

    print("\n" + "=" * 85)
    print("TABLE VII — Filter Performance: Detected vs Extrapolated Frames")
    print("=" * 85)
    print(f"{'Filter':<22} "
          f"{'Det MAE':>8} {'Det RMSE':>9} {'Det r':>7} {'Det N':>7}  "
          f"{'Ext MAE':>8} {'Ext RMSE':>9} {'Ext r':>7} {'Ext N':>7}")
    print("-" * 85)

    for fname in filter_names:
        d = detected_accum[fname]
        e = extrap_accum[fname]

        d_mae  = np.mean(d['MAE'])       if d['MAE']       else float('nan')
        d_rms  = np.mean(d['RMSE'])      if d['RMSE']      else float('nan')
        d_r    = np.mean(d['Pearson_r']) if d['Pearson_r'] else float('nan')
        d_n    = int(np.sum(d['n_frames'])) if d['n_frames'] else 0

        e_mae  = np.mean(e['MAE'])       if e['MAE']       else float('nan')
        e_rms  = np.mean(e['RMSE'])      if e['RMSE']      else float('nan')
        e_r    = np.mean(e['Pearson_r']) if e['Pearson_r'] else float('nan')
        e_n    = int(np.sum(e['n_frames'])) if e['n_frames'] else 0

        print(f"{labels[fname]:<22} "
              f"{d_mae:>8.4f} {d_rms:>9.4f} {d_r:>7.4f} {d_n:>7}  "
              f"{e_mae:>8.4f} {e_rms:>9.4f} {e_r:>7.4f} {e_n:>7}")

    print("=" * 85)
    print("  Det = camera-detected frames only")
    print("  Ext = SkillCorner-extrapolated frames only")
    print("  If Ext metrics better than Det: SkillCorner preprocessing")
    print("  aids filter performance on extrapolated frames.")