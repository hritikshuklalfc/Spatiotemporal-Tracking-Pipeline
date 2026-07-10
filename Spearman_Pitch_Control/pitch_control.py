import time

import numpy as np
from config import (pitch_len, pitch_width, grid_nx, grid_ny,
                    vmax, tr, beta,
                    home_team_id, away_team_id, min_played)

_gx = np.linspace(0, pitch_len, grid_nx)
_gy = np.linspace(0, pitch_width, grid_ny)
GX, GY = np.meshgrid(_gx, _gy, indexing='ij') 

def time_to_intercept(px, py, vx, vy):
    rx = px + vx * tr
    ry = py + vy * tr

    dist = np.sqrt((GX - rx) ** 2 + (GY - ry) ** 2)

    tau = tr + dist / vmax

    return np.maximum(tau, 1e-6)

def pitch_control_frame(home_players, away_players, beta_val):
    home_w = np.zeros((grid_nx, grid_ny))
    away_w = np.zeros((grid_nx, grid_ny))

    for p in home_players:
        tau     = time_to_intercept(p['px'], p['py'],
                                     p['vx'], p['vy'])
        home_w += tau ** (-beta_val)

    for p in away_players:
        tau     = time_to_intercept(p['px'], p['py'],
                                     p['vx'], p['vy'])
        away_w += tau ** (-beta_val)

    total = home_w + away_w
    total = np.where(total == 0, 1e-9, total)
    return home_w / total

def compute_mae_map(PC_filter, PC_star):
    return float(np.mean(np.abs(PC_filter - PC_star)))

def time_filter(filter_fn, x_noisy, y_noisy):
    start = time.perf_counter()
    filter_fn(x_noisy, y_noisy)
    return time.perf_counter() - start

def build_table_one(vel_cache, timing):
    filter_names = list(timing.keys())
    score_accum = {fname: {'RMSE': []} for fname in filter_names}

    print("\n" + "=" * 55)
    print("TABLE I — Velocity Recovery Error (RMSE)")
    print("=" * 55)
    print(f"{'Filter':<22} {'Mean RMSE':>12} {'Std':>10}")
    print("-" * 55)

    for pid, results in vel_cache.items():
        if results is None or 'ground_truth' not in results:
            continue

        vx_star, vy_star = results['ground_truth']
        v_star = np.sqrt(vx_star ** 2 + vy_star ** 2)

        for fname in filter_names:
            if fname not in results:
                continue

            vx_filtered, vy_filtered = results[fname]
            v_filtered = np.sqrt(vx_filtered ** 2 + vy_filtered ** 2)

            n = min(len(v_star), len(v_filtered))
            if n == 0:
                continue

            rmse = float(np.sqrt(np.mean((v_star[:n] - v_filtered[:n]) ** 2)))
            score_accum[fname]['RMSE'].append(rmse)

    label_map = {
        'moving_avg': 'Moving Average',
        'savitzky_golay': 'Savitzky-Golay',
        'butterworth': 'Butterworth (2Hz)',
        'kalman': 'Kalman (CV)',
    }

    for fname in filter_names:
        vals = score_accum[fname]['RMSE']
        if vals:
            print(f"{label_map.get(fname, fname):<22} {np.mean(vals):>12.4f} {np.std(vals):>10.4f}")
        else:
            print(f"{label_map.get(fname, fname):<22} {'N/A':>12}")

    print("=" * 55)
    return score_accum

def build_player_list(tracking, player_team, frame_idx,
                       filter_name, vel_cache):
    home_players = []
    away_players = []

    for pid, pdata in tracking.items():
        if pid not in vel_cache or vel_cache[pid] is None:
            continue
        if pid not in player_team:
            continue

        x_arr = np.array(pdata['x'])
        y_arr = np.array(pdata['y'])

        if frame_idx >= len(x_arr):
            continue
        px = float(x_arr[frame_idx])
        py = float(y_arr[frame_idx])
        if np.isnan(px) or np.isnan(py):
            continue

        vel_idx = frame_idx - 1
        if vel_idx < 0:
            continue

        vx_arr, vy_arr = vel_cache[pid][filter_name]
        if vel_idx >= len(vx_arr):
            continue

        vx = float(vx_arr[vel_idx])
        vy = float(vy_arr[vel_idx])

        if np.isnan(vx) or np.isnan(vy):
            vx, vy = 0.0, 0.0

        entry = {'px': px, 'py': py, 'vx': vx, 'vy': vy}

        if player_team[pid] == home_team_id:
            home_players.append(entry)
        elif player_team[pid] == away_team_id:
            away_players.append(entry)

    if len(home_players) < min_played or len(away_players) < min_played:
        return None, None

    return home_players, away_players