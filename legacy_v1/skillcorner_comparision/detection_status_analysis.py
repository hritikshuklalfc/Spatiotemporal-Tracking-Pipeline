"""Compare filter errors on SkillCorner detected and extrapolated frames."""

import json
from pathlib import Path
import sys

import numpy as np

# Reuse the established SkillCorner comparison filters and truth construction.
sys.path.insert(0, str(Path(__file__).parent))
import skillcorner_comarision as comparison


MATCH_IDS = [
    "1886347", "1899585", "1925299", "1953632", "1996435",
    "2006229", "2011166", "2013725", "2015213", "2017461",
]
BASE_DIR = Path(__file__).resolve().parent.parent / "data" / "archive_skillcorner"
FILTER_NAMES = ["moving_avg", "savitzky_golay", "butterworth", "kalman"]
LABELS = {
    "moving_avg": "Moving Average",
    "savitzky_golay": "Savitzky-Golay",
    "butterworth": "Butterworth (1.5Hz)",
    "kalman": "Kalman (CV)",
}

# Keep this analysis aligned with the current benchmark configuration.
comparison.ma_k = 7
comparison.sg_window = 15
comparison.sg_poly = 2
comparison.bw_fc = 1.5
comparison.bw_order = 2
comparison.kf_q_std = 0.01
comparison.kf_r_std = 1.5


def score_frames(x_method, y_method, x_truth, y_truth, is_detected, target_detected):
    """Score velocity transitions whose destination frame has the target flag."""
    vx_method = np.diff(x_method) / comparison.dt
    vy_method = np.diff(y_method) / comparison.dt
    vx_truth = np.diff(x_truth) / comparison.dt
    vy_truth = np.diff(y_truth) / comparison.dt

    method_speed = np.hypot(vx_method, vy_method)
    truth_speed = np.hypot(vx_truth, vy_truth)
    frame_mask = np.asarray(is_detected, dtype=bool)[1:] == target_detected
    valid = frame_mask & np.isfinite(method_speed) & np.isfinite(truth_speed)

    if not np.any(valid):
        return None

    error = method_speed[valid] - truth_speed[valid]
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "n_frames": int(np.sum(valid)),
    }


def run_match(match_id):
    tracking_path = BASE_DIR / match_id / f"{match_id}_tracking_extrapolated.jsonl"
    if not tracking_path.exists():
        print(f"[SKIP] {match_id}: tracking file not found")
        return None

    players = comparison.load_raw_tracking(tracking_path)
    totals = {
        state: {name: {"abs_error": [], "squared_error": [], "n_frames": 0}
                for name in FILTER_NAMES}
        for state in (True, False)
    }

    for pdata in players.values():
        x_raw = np.asarray(pdata["x"], dtype=float)
        y_raw = np.asarray(pdata["y"], dtype=float)
        is_detected = np.asarray(pdata["is_detected"], dtype=bool)
        if len(x_raw) < 20:
            continue

        x_truth, y_truth = comparison.build_linear_interpolation_truth(
            x_raw, y_raw, is_detected
        )
        filtered = comparison.apply_all_filter(x_raw, y_raw)

        for target_detected in (True, False):
            for name in FILTER_NAMES:
                x_filtered, y_filtered = filtered[name]
                vx_filtered = np.diff(x_filtered) / comparison.dt
                vy_filtered = np.diff(y_filtered) / comparison.dt
                vx_truth = np.diff(x_truth) / comparison.dt
                vy_truth = np.diff(y_truth) / comparison.dt
                method_speed = np.hypot(vx_filtered, vy_filtered)
                truth_speed = np.hypot(vx_truth, vy_truth)
                mask = (
                    (is_detected[1:] == target_detected)
                    & np.isfinite(method_speed)
                    & np.isfinite(truth_speed)
                )
                error = method_speed[mask] - truth_speed[mask]
                totals[target_detected][name]["abs_error"].extend(np.abs(error))
                totals[target_detected][name]["squared_error"].extend(error ** 2)
                totals[target_detected][name]["n_frames"] += int(np.sum(mask))

    print(f"\nMatch {match_id}")
    print(f"{'Frame status':<18} {'Filter':<22} {'MAE':>10} {'RMSE':>10} {'Frames':>10}")
    print("-" * 76)
    for target_detected in (True, False):
        status = "Detected (True)" if target_detected else "Extrapolated (False)"
        for name in FILTER_NAMES:
            values = totals[target_detected][name]
            if values["n_frames"]:
                mae = np.mean(values["abs_error"])
                rmse = np.sqrt(np.mean(values["squared_error"]))
                print(f"{status:<18} {LABELS[name]:<22} {mae:>10.4f} {rmse:>10.4f} {values['n_frames']:>10}")

    return totals


def print_summary(results):
    print("\n" + "=" * 76)
    print("ALL-MATCH SUMMARY")
    print("=" * 76)
    print(f"{'Frame status':<18} {'Filter':<22} {'MAE':>10} {'RMSE':>10} {'Frames':>10}")
    print("-" * 76)

    combined = {
        state: {name: {"abs_error": [], "squared_error": [], "n_frames": 0}
                for name in FILTER_NAMES}
        for state in (True, False)
    }
    for result in results:
        if result is None:
            continue
        for state in (True, False):
            for name in FILTER_NAMES:
                for key in ("abs_error", "squared_error"):
                    combined[state][name][key].extend(result[state][name][key])
                combined[state][name]["n_frames"] += result[state][name]["n_frames"]

    for state in (True, False):
        status = "Detected (True)" if state else "Extrapolated (False)"
        for name in FILTER_NAMES:
            values = combined[state][name]
            mae = np.mean(values["abs_error"])
            rmse = np.sqrt(np.mean(values["squared_error"]))
            print(f"{status:<18} {LABELS[name]:<22} {mae:>10.4f} {rmse:>10.4f} {values['n_frames']:>10}")


if __name__ == "__main__":
    outputs = [run_match(match_id) for match_id in MATCH_IDS]
    print_summary(outputs)
