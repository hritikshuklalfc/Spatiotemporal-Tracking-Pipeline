import json
import numpy as np
import os
from pathlib import Path

# ── You must fill in your 10 match IDs here ───────────────────────────────
MATCH_IDS = [
    "1886347",   # match 1 — already validated
    "1899585",   # match 2 — already validated
    "1925299",
    "1953632",
    "1996435",
    "2006229",
    "2011166",
    "2013725",
    "2015213",
    "2017461",
]

BASE_DIR    = "data/archive_skillcorner"
OUTPUTS_DIR = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Beta values for sensitivity analysis
BETA_VALUES = [2.3, 2.5, 2.7]
DEFAULT_BETA = 2.5

FILTER_NAMES = ['moving_avg', 'savitzky_golay', 'butterworth', 'kalman']
LABELS = {
    'moving_avg'     : 'Moving Average',
    'savitzky_golay' : 'Savitzky-Golay',
    'butterworth'    : 'Butterworth (2Hz)',
    'kalman'         : 'Kalman (CV)',
}


# ─────────────────────────────────────────────────────────────
# Per-match runner — calls your existing pipeline functions
# ─────────────────────────────────────────────────────────────

def run_single_match(match_id):
    
    # Build paths for this specific match
    match_json    = f"{BASE_DIR}/{match_id}/{match_id}_match.json"
    tracking_json = f"{BASE_DIR}/{match_id}/extracted_player_tracking.json"
    phases_csv    = f"{BASE_DIR}/{match_id}/{match_id}_phases_of_play.csv"

    # Verify files exist
    for path in [match_json, tracking_json, phases_csv]:
        if not Path(path).exists():
            print(f"  [SKIP] Missing: {path}")
            return None

    # Import raw loading functions that accept explicit paths
    import json
    import pandas as pd
    from filters import compute_all_filters, inject_noise, build_ground_truth
    from filters import (filter_moving_average, filter_savitzky_golay,
                         filter_butterworth, filter_kalman)
    from matrices import score_filter, time_filter
    from pitch_control import pitch_control_frame, compute_mae_map

    # ── Load match.json ───────────────────────────────────────
    with open(match_json) as f:
        match = json.load(f)

    home_team_id = match['home_team']['id']
    away_team_id = match['away_team']['id']

    player_team = {}
    for p in match['players']:
        player_team[str(p['id'])] = p['team_id']

    print(f"  [ok] {len(player_team)} players mapped, "
          f"home={home_team_id}, away={away_team_id}")

    # ── Load tracking ─────────────────────────────────────────
    with open(tracking_json) as f:
        tracking = json.load(f)

    # FIX: Check if the tracking file is empty to prevent an IndexError
    if not tracking:
        print(f"  [SKIP] Match {match_id} tracking file is empty or invalid.")
        return None

    # ── Load active frames ────────────────────────────────────
    from config import n_eval_frames, eval_period
    df_phases = pd.read_csv(phases_csv)
    df_phases = df_phases[df_phases['period'].astype(int) == eval_period]
    df_phases['frame_mid'] = (
        (df_phases['frame_start'].astype(int) +
         df_phases['frame_end'].astype(int)) / 2
    ).astype(int)
    candidates   = df_phases['frame_mid'].values
    idx          = np.linspace(0, len(candidates)-1,
                               n_eval_frames, dtype=int)
    active_frames = sorted(candidates[idx].tolist())

    # Build frame index from first player's frame list
    ref_pid    = list(tracking.keys())[0]
    ref_frames = tracking[ref_pid]['frame']
    frame_to_idx = {f: i for i, f in enumerate(ref_frames)}

    print(f"  [ok] {len(tracking)} players, "
          f"{len(active_frames)} eval frames selected")

    # ── Velocity computation ───────────────────────────────────
    vel_cache = {}
    for pid, pdata in tracking.items():
        vel_cache[pid] = compute_all_filters(pdata['x'], pdata['y'])

    n_valid = sum(1 for v in vel_cache.values() if v is not None)
    print(f"  [ok] {n_valid}/{len(tracking)} players have valid output")

    # ── Timing ────────────────────────────────────────────────
    ref_x = np.array(tracking[ref_pid]['x'], dtype=float)
    ref_y = np.array(tracking[ref_pid]['y'], dtype=float)
    x_star_ref, y_star_ref, _, _ = build_ground_truth(ref_x, ref_y)
    x_noisy_ref, y_noisy_ref     = inject_noise(x_star_ref, y_star_ref)

    timing = {
        'moving_avg'     : time_filter(filter_moving_average,
                                        x_noisy_ref, y_noisy_ref),
        'savitzky_golay' : time_filter(filter_savitzky_golay,
                                        x_noisy_ref, y_noisy_ref),
        'butterworth'    : time_filter(filter_butterworth,
                                        x_noisy_ref, y_noisy_ref),
        'kalman'         : time_filter(filter_kalman,
                                        x_noisy_ref, y_noisy_ref),
    }

    # ── Velocity metrics ──────────────────────────────────────
    vel_metrics = {f: {'MAE': [], 'RMSE': [], 'Pearson_r': []}
                   for f in FILTER_NAMES}

    for pid, result in vel_cache.items():
        if result is None:
            continue
        for fname in FILTER_NAMES:
            s = score_filter(result[fname], result['ground_truth'])
            for metric, val in s.items():
                if np.isfinite(val):
                    vel_metrics[fname][metric].append(val)

    # ── Pitch control ─────────────────────────────────────────
    pc_metrics = {b: {f: [] for f in FILTER_NAMES}
                  for b in BETA_VALUES}
    frames_evaluated = 0

    for frame_num in active_frames:
        if frame_num not in frame_to_idx:
            continue
        fidx = frame_to_idx[frame_num]

        for beta_val in BETA_VALUES:
            # Build player lists using this match's team IDs
            home_gt, away_gt = _build_player_list(
                tracking, player_team,
                home_team_id, away_team_id,
                fidx, 'ground_truth', vel_cache)

            if home_gt is None:
                continue

            PC_star = pitch_control_frame(home_gt, away_gt, beta_val)

            for fname in FILTER_NAMES:
                home_f, away_f = _build_player_list(
                    tracking, player_team,
                    home_team_id, away_team_id,
                    fidx, fname, vel_cache)
                if home_f is None:
                    continue
                PC_f = pitch_control_frame(home_f, away_f, beta_val)
                pc_metrics[beta_val][fname].append(
                    compute_mae_map(PC_f, PC_star))

        frames_evaluated += 1

    print(f"  [ok] {frames_evaluated} frames evaluated for pitch control")

    # ── Package and return ────────────────────────────────────
    return {
        'match_id'  : match_id,
        'n_players' : n_valid,
        'n_frames'  : frames_evaluated,
        'timing'    : timing,
        'vel_metrics': {
            fname: {
                metric: float(np.mean(vals)) if vals else None
                for metric, vals in metrics.items()
            }
            for fname, metrics in vel_metrics.items()
        },
        'pc_metrics': {
            str(b): {
                fname: {
                    'mean': float(np.mean(vals)) if vals else None,
                    'std' : float(np.std(vals))  if vals else None,
                }
                for fname, vals in beta_results.items()
            }
            for b, beta_results in pc_metrics.items()
        }
    }


def _build_player_list(tracking, player_team,
                        home_team_id, away_team_id,
                        frame_idx, filter_name, vel_cache,
                        min_players=5):
    """
    Same logic as pitch_control.build_player_list but takes
    explicit home/away team IDs instead of reading from config.
    This is necessary because each match has different team IDs.
    """
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

    if len(home_players) < min_players or len(away_players) < min_players:
        return None, None

    return home_players, away_players


# ─────────────────────────────────────────────────────────────
# Cross-match aggregation and printing
# ─────────────────────────────────────────────────────────────

def print_cross_match_tables(all_results):
    """
    Takes list of per-match result dicts and prints:
      Table IV  — velocity metrics across matches (mean ± std)
      Table V   — pitch control distortion across matches
      Table VI  — ranking consistency check
    """
    valid = [r for r in all_results if r is not None]
    n     = len(valid)

    if n == 0:
        print("No valid match results to aggregate.")
        return

    print(f"\n\nAggregating results across {n} matches...\n")

    # ── TABLE IV: Velocity metrics ─────────────────────────────
    print("=" * 75)
    print(f"TABLE IV — Cross-Match Velocity Recovery Error (N={n} matches)")
    print("=" * 75)
    print(f"{'Filter':<22} {'MAE mean':>10} {'MAE std':>8} "
          f"{'RMSE mean':>10} {'RMSE std':>8} {'Pearson r':>10}")
    print("-" * 75)

    for fname in FILTER_NAMES:
        maes  = [r['vel_metrics'][fname]['MAE']       for r in valid
                 if r['vel_metrics'][fname]['MAE'] is not None]
        rmses = [r['vel_metrics'][fname]['RMSE']      for r in valid
                 if r['vel_metrics'][fname]['RMSE'] is not None]
        rs    = [r['vel_metrics'][fname]['Pearson_r'] for r in valid
                 if r['vel_metrics'][fname]['Pearson_r'] is not None]

        print(f"{LABELS[fname]:<22} "
              f"{np.mean(maes):>10.4f} {np.std(maes):>8.4f} "
              f"{np.mean(rmses):>10.4f} {np.std(rmses):>8.4f} "
              f"{np.mean(rs):>10.4f}")
    print("=" * 75)

    # ── TABLE V: Pitch control at default beta ─────────────────
    b_key = str(DEFAULT_BETA)
    print(f"\n{'='*65}")
    print(f"TABLE V — Cross-Match MAEmap at β={DEFAULT_BETA} (N={n} matches)")
    print(f"{'='*65}")
    print(f"{'Filter':<22} {'Mean MAEmap':>12} {'Std (within)':>14} "
          f"{'Std (across)':>14}")
    print("-" * 65)

    for fname in FILTER_NAMES:
        within_means = [r['pc_metrics'][b_key][fname]['mean'] for r in valid
                        if r['pc_metrics'][b_key][fname]['mean'] is not None]
        within_stds  = [r['pc_metrics'][b_key][fname]['std']  for r in valid
                        if r['pc_metrics'][b_key][fname]['std']  is not None]

        if within_means:
            print(f"{LABELS[fname]:<22} "
                  f"{np.mean(within_means):>12.4f} "
                  f"{np.mean(within_stds):>14.4f} "
                  f"{np.std(within_means):>14.4f}")
    print("=" * 65)
    print("  Std (within)  = average frame-to-frame variance within a match")
    print("  Std (across)  = variance of mean MAEmap across matches")
    print("  If std(across) << std(within): filter effect is match-independent")

    # ── TABLE VI: Ranking consistency ─────────────────────────
    print(f"\n{'='*65}")
    print(f"TABLE VI — MAEmap Ranking Consistency (N={n} matches)")
    print(f"{'='*65}")
    print(f"  Rank 1 = lowest MAEmap (best), Rank 4 = highest (worst)")
    print(f"{'Filter':<22} {'Rank per match':<30} {'Always same rank?':>18}")
    print("-" * 65)

    # Compute ranking per match
    for fname in FILTER_NAMES:
        ranks = []
        for r in valid:
            match_means = {
                f: r['pc_metrics'][b_key][f]['mean']
                for f in FILTER_NAMES
                if r['pc_metrics'][b_key][f]['mean'] is not None
            }
            sorted_filters = sorted(match_means, key=match_means.get)
            rank = sorted_filters.index(fname) + 1
            ranks.append(str(rank))

        rank_str    = '  '.join(ranks)
        is_stable   = len(set(ranks)) == 1
        stability   = "YES ✓" if is_stable else f"NO  (varies: {set(ranks)})"
        print(f"{LABELS[fname]:<22} {rank_str:<30} {stability:>18}")
    print("=" * 65)


def save_results(all_results):
    """Save all match results to JSON for later analysis."""
    path = os.path.join(OUTPUTS_DIR, "cross_match_results.json")
    with open(path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results saved to: {path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    all_results = []

    for match_id in MATCH_IDS:
        result = run_single_match(match_id)
        all_results.append(result)

        # Save after every match in case of interruption
        save_results(all_results)

    print_cross_match_tables(all_results)

    # Count successes
    n_ok   = sum(1 for r in all_results if r is not None)
    n_fail = sum(1 for r in all_results if r is None)
    print(f"\nCompleted: {n_ok}/{len(MATCH_IDS)} matches processed successfully")
    if n_fail > 0:
        print(f"Failed: {n_fail} matches — check error messages above")


if __name__ == "__main__":
    main()