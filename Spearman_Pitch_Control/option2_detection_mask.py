import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

from filters import compute_all_filters

MATCH_IDS = [
    "1886347", "1899585", "1925299", "1953632", "1996435",
    "2006229", "2011166", "2013725", "2015213", "2017461",
]
BASE_DIR     = "data/archive_skillcorner"
FILTER_NAMES = ['moving_avg', 'savitzky_golay', 'butterworth', 'kalman']
LABELS = {
    'moving_avg'     : 'Moving Average',
    'savitzky_golay' : 'Savitzky-Golay',
    'butterworth'    : 'Butterworth (2Hz)',
    'kalman'         : 'Kalman (CV)',
}


# ─────────────────────────────────────────────────────────────
# Metrics — defined locally to avoid any import confusion
# ─────────────────────────────────────────────────────────────

def _to_speed(v):
    return np.sqrt(np.array(v[0])**2 + np.array(v[1])**2)


def _score_on_mask(v_hat, v_star, mask):
    # Convert to speed FIRST, then apply mask
    speed_hat  = _to_speed(v_hat)
    speed_star = _to_speed(v_star)

    # Align all lengths
    n = min(len(speed_hat), len(speed_star), len(mask))
    speed_hat  = speed_hat[:n]
    speed_star = speed_star[:n]
    mask       = np.array(mask[:n], dtype=bool)

    # Apply mask
    sh_masked = speed_hat[mask]
    ss_masked = speed_star[mask]

    n_frames = int(mask.sum())
    if n_frames < 10:
        return None

    mae_val  = float(np.mean(np.abs(sh_masked - ss_masked)))
    rmse_val = float(np.sqrt(np.mean((sh_masked - ss_masked)**2)))

    if np.std(sh_masked) < 1e-9 or np.std(ss_masked) < 1e-9:
        r_val = 0.0
    else:
        r_val = float(pearsonr(sh_masked, ss_masked)[0])

    return {
        'MAE'      : mae_val,
        'RMSE'     : rmse_val,
        'Pearson_r': r_val,
        'n_frames' : n_frames,
    }


# ─────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────

def run_detection_mask_analysis():

    detected_accum = {f: {'MAE': [], 'RMSE': [], 'Pearson_r': [], 'n_frames': []}
                      for f in FILTER_NAMES}
    extrap_accum   = {f: {'MAE': [], 'RMSE': [], 'Pearson_r': [], 'n_frames': []}
                      for f in FILTER_NAMES}

    match_det_rates = {}

    for match_id in MATCH_IDS:
        tracking_path = f"{BASE_DIR}/{match_id}/extracted_player_tracking.json"
        mask_path     = f"{BASE_DIR}/{match_id}/detection_mask.json"

        if not Path(tracking_path).exists():
            print(f"[SKIP] {match_id} — no tracking file")
            continue
        if not Path(mask_path).exists():
            print(f"[SKIP] {match_id} — no detection mask")
            continue

        with open(tracking_path) as f:
            tracking = json.load(f)
        with open(mask_path) as f:
            detection_masks = json.load(f)

        track_ids  = set(tracking.keys())
        mask_ids   = set(detection_masks.keys())
        overlap    = track_ids & mask_ids
        only_track = track_ids - mask_ids
        only_mask  = mask_ids  - track_ids

        print(f"\nMatch {match_id}:")
        print(f"  Tracking players:      {len(track_ids)}")
        print(f"  Detection mask players:{len(mask_ids)}")
        print(f"  Overlap:               {len(overlap)}")
        if only_track:
            print(f"  Only in tracking:      {list(only_track)[:3]}")
        if only_mask:
            print(f"  Only in mask:          {list(only_mask)[:3]}")
        if len(overlap) == 0:
            print(f"  [WARNING] Zero overlap — skipping match {match_id}")
            continue

        all_detected_flags = []
        players_matched    = 0
        players_skipped    = 0

        for pid in overlap:
            pdata    = tracking[pid]
            mask_data = detection_masks[pid]

            result = compute_all_filters(pdata['x'], pdata['y'])
            if result is None:
                players_skipped += 1
                continue

            # Build detection boolean array
            is_detected = np.array(mask_data['is_detected'], dtype=bool)
            all_detected_flags.extend(is_detected.tolist())
            v_len = len(result['ground_truth'][0])

            if len(is_detected) < 2:
                players_skipped += 1
                continue

            # Align: take min of available mask and velocity length
            align_len    = min(len(is_detected) - 1, v_len)
            det_mask     = is_detected[1:align_len+1]     # detected frames
            extrap_mask  = ~det_mask                       # extrapolated frames

            # Debug: check mask content
            n_det    = det_mask.sum()
            n_extrap = extrap_mask.sum()

            if players_matched == 0:  # print once per match
                print(f"  Sample player {pid}: "
                      f"{n_det} detected frames, "
                      f"{n_extrap} extrapolated frames "
                      f"(vel_len={v_len}, mask_len={len(is_detected)})")

            v_star = result['ground_truth']

            for fname in FILTER_NAMES:
                v_hat = result[fname]

                # Score on detected frames
                s_det = _score_on_mask(v_hat, v_star, det_mask)
                if s_det is not None:
                    for metric in ['MAE', 'RMSE', 'Pearson_r']:
                        if np.isfinite(s_det[metric]):
                            detected_accum[fname][metric].append(s_det[metric])
                    detected_accum[fname]['n_frames'].append(s_det['n_frames'])

                # Score on extrapolated frames
                s_ext = _score_on_mask(v_hat, v_star, extrap_mask)
                if s_ext is not None:
                    for metric in ['MAE', 'RMSE', 'Pearson_r']:
                        if np.isfinite(s_ext[metric]):
                            extrap_accum[fname][metric].append(s_ext[metric])
                    extrap_accum[fname]['n_frames'].append(s_ext['n_frames'])

            players_matched += 1

        det_rate = np.mean(all_detected_flags) * 100 if all_detected_flags else 0
        match_det_rates[match_id] = det_rate
        print(f"  Players matched: {players_matched}, skipped: {players_skipped}")
        print(f"  Detection rate: {det_rate:.1f}%")

    # ── Print summary detection stats ──────────────────────────
    rates = list(match_det_rates.values())
    print(f"\n{'='*60}")
    print(f"Detection Statistics (N={len(rates)} matches)")
    print(f"  Mean: {np.mean(rates):.1f}%  "
          f"Std: {np.std(rates):.1f}%  "
          f"Range: {np.min(rates):.1f}–{np.max(rates):.1f}%")

    # ── TABLE VII ─────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("TABLE VII — Filter Performance: Detected vs Extrapolated Frames")
    print(f"{'='*90}")
    print(f"{'Filter':<22} "
          f"{'Det MAE':>8} {'Det RMSE':>9} {'Det r':>7} {'Det N':>8}  "
          f"{'Ext MAE':>8} {'Ext RMSE':>9} {'Ext r':>7} {'Ext N':>8}")
    print("-" * 90)

    for fname in FILTER_NAMES:
        d = detected_accum[fname]
        e = extrap_accum[fname]

        d_mae = np.mean(d['MAE'])       if d['MAE']       else float('nan')
        d_rms = np.mean(d['RMSE'])      if d['RMSE']      else float('nan')
        d_r   = np.mean(d['Pearson_r']) if d['Pearson_r'] else float('nan')
        d_n   = int(np.sum(d['n_frames'])) if d['n_frames'] else 0

        e_mae = np.mean(e['MAE'])       if e['MAE']       else float('nan')
        e_rms = np.mean(e['RMSE'])      if e['RMSE']      else float('nan')
        e_r   = np.mean(e['Pearson_r']) if e['Pearson_r'] else float('nan')
        e_n   = int(np.sum(e['n_frames'])) if e['n_frames'] else 0

        print(f"{LABELS[fname]:<22} "
              f"{d_mae:>8.4f} {d_rms:>9.4f} {d_r:>7.4f} {d_n:>8}  "
              f"{e_mae:>8.4f} {e_rms:>9.4f} {e_r:>7.4f} {e_n:>8}")

    print("=" * 90)
    print("  Det = camera-detected frames only")
    print("  Ext = SkillCorner-extrapolated frames only")

    # ── KEY RATIO TABLE ───────────────────────────────────────
    print(f"\n{'='*75}")
    print("KEY COMPARISON — Extrapolated vs Detected MAE ratio")
    print(f"{'='*75}")
    print(f"{'Filter':<22} {'Det MAE':>10} {'Ext MAE':>10} "
          f"{'Ratio':>8} {'Interpretation':>20}")
    print("-" * 75)

    for fname in FILTER_NAMES:
        d_mae = np.mean(detected_accum[fname]['MAE']) \
                if detected_accum[fname]['MAE'] else float('nan')
        e_mae = np.mean(extrap_accum[fname]['MAE'])   \
                if extrap_accum[fname]['MAE']   else float('nan')
        ratio = e_mae / d_mae if (np.isfinite(d_mae) and d_mae > 0
                                   and np.isfinite(e_mae)) else float('nan')

        if not np.isfinite(ratio):
            interp = "No data"
        elif ratio < 0.9:
            interp = "Ext easier ✓"
        elif ratio > 1.1:
            interp = "Ext harder ✗"
        else:
            interp = "No difference"

        print(f"{LABELS[fname]:<22} {d_mae:>10.4f} {e_mae:>10.4f} "
              f"{ratio:>8.3f} {interp:>20}")

    print("=" * 75)
    print("\nRatio < 0.9 → SkillCorner preprocessing helps your filters")
    print("Ratio > 1.1 → Model-generated positions harder to filter")
    print("Ratio 0.9–1.1 → Detection status has no meaningful effect")


if __name__ == "__main__":
    run_detection_mask_analysis()