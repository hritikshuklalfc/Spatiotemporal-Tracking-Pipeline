

import numpy as np
import time

# ── Local modules ──────────────────────────────────────────────────────────
from config       import beta
from loader        import (load_player_team_map, load_tracking,
                            load_active_frame, build_frame_index)
from filters       import compute_all_filters, inject_noise, build_ground_truth
from filters       import (filter_moving_average, filter_savitzky_golay,
                            filter_butterworth, filter_kalman)
from pitch_control import time_filter, build_table_one
from pitch_control import (pitch_control_frame, compute_mae_map,
                            build_player_list)
from visualiser     import (plot_fig1_velocity_comparison,
                            plot_fig2_pitch_control,
                            plot_difference_map,
                            plot_pareto_scatter)


def main():

    # ──────────────────────────────────────────────────────────
    # STEP 1 — Load data
    # ──────────────────────────────────────────────────────────
    print("\n── STEP 1: Loading data ─────────────────────────────")
    player_team   = load_player_team_map()
    tracking      = load_tracking()
    active_frames = load_active_frame()
    frame_to_idx  = build_frame_index(tracking)


    # ──────────────────────────────────────────────────────────
    # STEP 2 — Compute velocities for every player, all filters
    # ──────────────────────────────────────────────────────────
    print("\n── STEP 2: Computing velocities (all filters) ───────")
    vel_cache = {}
    for pid, pdata in tracking.items():
        vel_cache[pid] = compute_all_filters(pdata['x'], pdata['y'])

    n_valid = sum(1 for v in vel_cache.values() if v is not None)
    print(f"[main] {n_valid}/{len(tracking)} players have valid filter output")


    # ──────────────────────────────────────────────────────────
    # STEP 3 — Time each filter (wall-clock, Table I column 4)
    # ──────────────────────────────────────────────────────────
    print("\n── STEP 3: Timing filters ───────────────────────────")

    # Use the first valid player's noisy signal as timing input
    ref_pid = next(pid for pid, v in vel_cache.items() if v is not None)
    ref_x   = np.array(tracking[ref_pid]['x'], dtype=float)
    ref_y   = np.array(tracking[ref_pid]['y'], dtype=float)
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
    for fname, t in timing.items():
        print(f"  {fname:<20} {t:.4f} s")


    # ──────────────────────────────────────────────────────────
    # STEP 4 — TABLE I: velocity recovery error
    # ──────────────────────────────────────────────────────────
    print("\n── STEP 4: Table I — Velocity Recovery Error ────────")
    score_accum = build_table_one(vel_cache, timing)


    # ──────────────────────────────────────────────────────────
    # STEP 5 — FIGURE 1: raw vs filtered vs ground truth
    # ──────────────────────────────────────────────────────────
    print("\n── STEP 5: Figure 1 ─────────────────────────────────")

    ref_result = vel_cache[ref_pid]
    vx_raw,  vy_raw  = ref_result['raw_noisy']
    vx_sg,   vy_sg   = ref_result['savitzky_golay']
    vx_star, vy_star = ref_result['ground_truth']

    # Speed magnitude for 1D plot
    v_raw_spd  = np.sqrt(vx_raw**2  + vy_raw**2)
    v_sg_spd   = np.sqrt(vx_sg**2   + vy_sg**2)
    v_star_spd = np.sqrt(vx_star**2 + vy_star**2)

    # Show a 5-minute window (300 frames at 10Hz)
    window = slice(0, min(3000, len(v_raw_spd)))
    plot_fig1_velocity_comparison(
        v_raw_spd[window],
        v_sg_spd[window],
        v_star_spd[window],
        filter_label="Savitzky-Golay",
        player_id=ref_pid
    )


    # ──────────────────────────────────────────────────────────
    # STEP 6 — TABLE II + TABLE III: pitch control distortion
    #           Table II  = results at default beta (2.5)
    #           Table III = sensitivity across all three beta values
    # ──────────────────────────────────────────────────────────
    print("\n── STEP 6: Pitch Control Distortion ─────────────────")

    FILTER_NAMES = ['moving_avg', 'savitzky_golay',
                    'butterworth', 'kalman']
    LABELS = {
        'moving_avg'     : 'Moving Average',
        'savitzky_golay' : 'Savitzky-Golay',
        'butterworth'    : 'Butterworth (2Hz)',
        'kalman'         : 'Kalman (CV)',
    }

    # beta is a list e.g. [2.3, 2.5, 2.7] — imported from config
    # We store results per beta value
    all_beta_results = {}   # {beta_val -> {filter_name -> [mae values]}}

    PC_star_last   = None
    PC_filter_last = {f: None for f in FILTER_NAMES}

    for beta_val in beta:
        mae_map_accum = {f: [] for f in FILTER_NAMES}

        for frame_num in active_frames:
            if frame_num not in frame_to_idx:
                continue
            fidx = frame_to_idx[frame_num]

            # Ground truth surface at this beta value
            h_gt, a_gt = build_player_list(
                tracking, player_team, fidx, 'ground_truth', vel_cache)
            if h_gt is None:
                continue

            # Pass beta_val explicitly — single float
            PC_star = pitch_control_frame(h_gt, a_gt, beta_val)
            PC_star_last = PC_star

            for fname in FILTER_NAMES:
                h_f, a_f = build_player_list(
                    tracking, player_team, fidx, fname, vel_cache)
                if h_f is None:
                    continue
                PC_f = pitch_control_frame(h_f, a_f, beta_val)
                mae  = compute_mae_map(PC_f, PC_star)
                mae_map_accum[fname].append(mae)
                PC_filter_last[fname] = PC_f

        all_beta_results[beta_val] = mae_map_accum

    # ── Print Table II — default beta only (2.5) ──────────────
    default_beta = 2.5
    print("\n" + "=" * 55)
    print(f"TABLE II — Pitch-Control Distortion (β = {default_beta})")
    print("=" * 55)
    print(f"{'Filter':<22} {'Mean MAEmap':>12} {'Std':>10}")
    print("-" * 55)
    mae_map_means = {}
    for fname in FILTER_NAMES:
        vals = all_beta_results[default_beta][fname]
        if vals:
            m = np.mean(vals)
            s = np.std(vals)
            mae_map_means[fname] = m
            print(f"{LABELS[fname]:<22} {m:>12.4f} {s:>10.4f}")
        else:
            print(f"{LABELS[fname]:<22} {'N/A':>12}")
    print("=" * 55)

    # ── Print Table III — beta sensitivity ────────────────────
    print("\n" + "=" * 70)
    print("TABLE III — MAEmap Sensitivity to β")
    print("=" * 70)
    header = f"{'Filter':<22}"
    for b in beta:
        header += f"  β={b}"
    print(header)
    print("-" * 70)
    for fname in FILTER_NAMES:
        row = f"{LABELS[fname]:<22}"
        for b in beta:
            vals = all_beta_results[b][fname]
            if vals:
                row += f"  {np.mean(vals):.4f}"
            else:
                row += f"  {'N/A':>6}"
        print(row)
    print("=" * 70)

    # ──────────────────────────────────────────────────────────
    # STEP 7 — FIGURE 2: pitch control surface
    # ──────────────────────────────────────────────────────────
    print("\n── STEP 7: Figure 2 ─────────────────────────────────")
    if PC_star_last is not None:
        plot_fig2_pitch_control(
            PC_star_last,
            title="Fig 2 — Pitch Control Surface (Ground Truth Velocities)"
        )


    # ──────────────────────────────────────────────────────────
    # STEP 8 — FIGURE 3: difference map
    # ──────────────────────────────────────────────────────────
    print("\n── STEP 8: Figure 3 ─────────────────────────────────")

    if mae_map_means:
        best_filter  = min(mae_map_means, key=mae_map_means.get)
        worst_filter = max(mae_map_means, key=mae_map_means.get)

        PC_best  = PC_filter_last[best_filter]
        PC_worst = PC_filter_last[worst_filter]

        if PC_best is not None and PC_worst is not None:
            plot_difference_map(
                PC_best, PC_worst,
                label_a=LABELS[best_filter],
                label_b=LABELS[worst_filter]
            )
        print(f"[main] Best filter:  {LABELS[best_filter]} "
              f"(MAEmap={mae_map_means[best_filter]:.4f})")
        print(f"[main] Worst filter: {LABELS[worst_filter]} "
              f"(MAEmap={mae_map_means[worst_filter]:.4f})")

    # ──────────────────────────────────────────────────────────
    # STEP 9 — FIGURE 4: Pareto scatter
    # ──────────────────────────────────────────────────────────
    print("\n── STEP 9: Figure 4 — Pareto scatter ────────────────")

    # Pull RMSE means from score_accum
    rmse_means = {}
    for fname in FILTER_NAMES:
        vals = score_accum[fname]['RMSE']
        if vals:
            rmse_means[fname] = np.mean(vals)

    plot_pareto_scatter(rmse_means, timing)

    print("\n── DONE ─────────────────────────────────────────────")
    print("All tables printed above. Figures saved to outputs/")
    print("Fill Table I and Table II in your paper with the numbers above.")


if __name__ == "__main__":
    main()