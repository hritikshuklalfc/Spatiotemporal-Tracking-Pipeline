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


def score_filter(v_hat, v_star):
    speed_hat  = np.sqrt(v_hat[0]**2 + v_hat[1]**2)
    speed_star = np.sqrt(v_star[0]**2 + v_star[1]**2)

    return {
        'MAE'      : mae(speed_hat,      speed_star),
        'RMSE'     : rmse(speed_hat,     speed_star),
        'Pearson_r': pearson_r(speed_hat, speed_star),
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
        'butterworth'    : 'Butterworth (2Hz)',
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