#!/usr/bin/env python3
"""
benchmark_v2_pitchcontrol.py -- downstream pitch-control distortion (MAE_map), corrected
=======================================================================================
Run from the REPOSITORY ROOT, AFTER benchmark_v2.py (it reads the tuned parameters
that script wrote to <tuning-dir>/tuning_<gt>_<noise>.csv):

    python Spearman_Pitch_Control/benchmark_v2.py --gt sg15            # writes outputs/v2/tuning_*.csv
    python Spearman_Pitch_Control/benchmark_v2_pitchcontrol.py --gt sg15

Why a rewrite was needed (v1 problems, verified on the v1 code and data):
  A. FRAME ALIGNMENT. v1 looked up the evaluation frame in the FIRST player's frame
     list and then used that same array index for EVERY player. But 14 of 29 players
     in a match have different frame lists (they are missing different stretches), so
     the "frame" mixed positions from different moments of the match. Here every
     player is looked up by the true frame number inside its own contiguous segment.
  B. GAPS. Series are split at frame-number gaps (see benchmark_v2.py), so velocities
     are never computed across a gap.
  C. TUNING. All filters use parameters tuned on the tuning matches only; scores are on
     the test matches.
  D. BASELINES. Two baselines were missing: RAW (velocity from the unfiltered noisy
     positions) and STATIC (all velocities set to zero, i.e. ignore velocity). STATIC
     answers "how much does ANY velocity estimate add to this pitch-control model?".
  F. COORDINATES. Tracking positions are centred on the pitch centre (x about -52..52,
     y about -34..34) while the pitch-control grid spans 0..104 x 0..68, and v1 never
     shifted them, so every player was displaced by (-52, -34) m relative to the grid.
     Positions are shifted to grid coordinates here.
  E. MORE FRAMES. v1 used 25 frames per match, and its own table showed the
     within-match spread was larger than the mean. Default here: 200 frames per match,
     spread evenly over both halves.

Everything else is unchanged from v1: same Spearman model (pitch_control.py, with
its config values vmax, tr, grid), same MAE_map definition, positions are the observed
positions and only the velocity vectors differ between methods.
"""
import argparse
import json
import os
import time
import numpy as np
import pandas as pd

import benchmark_v2 as B
from pitch_control import pitch_control_frame, compute_mae_map
from config import pitch_len, pitch_width

# Tracking coordinates are centred on the pitch centre (x ~ -52..52, y ~ -34..34) but the
# pitch-control grid spans 0..pitch_len x 0..pitch_width. v1 never converted between them.
OFF_X, OFF_Y = pitch_len / 2.0, pitch_width / 2.0


def load_selected(tuning_dir, gt, noise):
    path = os.path.join(tuning_dir, f"tuning_{gt}_{noise}.csv")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}: run benchmark_v2.py --gt {gt} first")
    t = pd.read_csv(path)
    methods = {}
    for pre in ("none", "hampel"):
        for fam in ("MA", "SG", "BW", "KF"):
            s = t[(t.pre == pre) & (t.family == fam)].sort_values("rmse")
            if len(s):
                tag = "" if pre == "none" else "Hampel+"
                methods[f"{tag}{fam}(tuned)"] = (fam, json.loads(s.iloc[0].params), pre)
    for fam, p in B.PAPER_DEFAULTS.items():
        methods[f"{fam}(paper-default)"] = (fam, p, "none")
    return methods


def select_frames(segs, team_of, n_frames, step=25, min_per_team=8):
    """Frames where at least `min_per_team` players per team are inside an interior part
    of a segment; n_frames of them chosen evenly across the match."""
    f0 = np.array([s[2] for s in segs])
    n = np.array([len(s[3]) for s in segs])
    team = np.array([team_of.get(str(s[0]), -1) for s in segs])
    cands = np.arange(f0.min() + B.TRIM + 1, (f0 + n).max(), step)
    good = []
    for f in cands:
        m = (f >= f0 + B.TRIM + 1) & (f <= f0 + n - 1 - B.TRIM)
        t = team[m]
        if (t == team_of["home"]).sum() >= min_per_team and (t == team_of["away"]).sum() >= min_per_team:
            good.append(int(f))
    if not good:
        return []
    idx = np.unique(np.linspace(0, len(good) - 1, min(n_frames, len(good))).astype(int))
    return [good[i] for i in idx]


def run_match(data_root, mid, gt_kind, noise_name, methods, n_frames, beta):
    with open(os.path.join(data_root, mid, f"{mid}_match.json")) as f:
        match = json.load(f)
    team_of = {str(p["id"]): p["team_id"] for p in match["players"]}
    team_of["home"], team_of["away"] = match["home_team"]["id"], match["away_team"]["id"]
    segs = B.load_match(data_root, mid)
    frames = select_frames(segs, team_of, n_frames)
    nfn, tag = B.NOISE[noise_name]
    # frame -> list of player records
    recs = {f: [] for f in frames}
    fr_arr = np.array(frames)
    for si_all, (pid, si, f0, x, y) in enumerate(segs):
        team = team_of.get(str(pid))
        if team not in (team_of["home"], team_of["away"]):
            continue
        n = len(x)
        inside = fr_arr[(fr_arr >= f0 + B.TRIM + 1) & (fr_arr <= f0 + n - 1 - B.TRIM)]
        if len(inside) == 0:
            continue
        xs, ys = B.gt_positions(x, y, gt_kind)
        vg = B.speed(xs, ys)
        ok = B.valid_mask(vg)
        xn, yn = nfn(xs.copy(), ys.copy(), B.make_rng(mid, si_all, tag))   # same seeds as benchmark_v2.py
        vel = {"RAW(unfiltered)": (np.diff(xn) / B.DT, np.diff(yn) / B.DT)}
        for name, (fam, p, pre) in methods.items():
            xf, yf = B.apply_cfg(xn, yn, fam, p, pre)
            vel[name] = (np.diff(xf) / B.DT, np.diff(yf) / B.DT)
        gvx, gvy = np.diff(xs) / B.DT, np.diff(ys) / B.DT
        for f in inside:
            i = int(f - f0)                 # position index; velocity index is i-1 (as in v1)
            if not ok[i - 1]:
                continue
            recs[int(f)].append(dict(team=team, px=float(x[i]) + OFF_X, py=float(y[i]) + OFF_Y,
                                     gt=(float(gvx[i - 1]), float(gvy[i - 1])),
                                     vel={k: (float(v[0][i - 1]), float(v[1][i - 1])) for k, v in vel.items()}))
    rows = []
    names = ["RAW(unfiltered)"] + list(methods.keys()) + ["STATIC(zero velocity)"]
    for f in frames:
        rs = recs[f]
        home = [r for r in rs if r["team"] == team_of["home"]]
        away = [r for r in rs if r["team"] == team_of["away"]]
        if len(home) < 5 or len(away) < 5:
            continue

        def build(sel):
            h = [dict(px=r["px"], py=r["py"], vx=sel(r)[0], vy=sel(r)[1]) for r in home]
            a = [dict(px=r["px"], py=r["py"], vx=sel(r)[0], vy=sel(r)[1]) for r in away]
            return pitch_control_frame(h, a, beta)
        pc_star = build(lambda r: r["gt"])
        for name in names:
            if name == "STATIC(zero velocity)":
                pc = build(lambda r: (0.0, 0.0))
            else:
                pc = build(lambda r, nm=name: r["vel"][nm])
            rows.append((mid, f, name, compute_mae_map(pc, pc_star), len(home), len(away)))
    return rows


def paired(pm, a, b, rng, nboot=2000):
    """Per-match difference a-b (negative = a has LOWER distortion) with a match-level bootstrap CI."""
    p = pm.pivot(index="match", columns="method", values="mae_map")
    if a not in p or b not in p:
        return None
    d = (p[a] - p[b]).values
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(nboot)]
    return d.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5), int((d < 0).sum()), len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="sg15", choices=["sg15", "gauss"])
    ap.add_argument("--noise", default="N1,N2")
    ap.add_argument("--tuning-dir", default="outputs/v2")
    ap.add_argument("--data", default="data/archive_skillcorner")
    ap.add_argument("--out", default="outputs/v2")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--beta", type=float, default=2.5)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    log = []

    def emit(s=""):
        print(s, flush=True)
        log.append(str(s))

    tests = B.TEST_MATCHES[:1] if a.quick else B.TEST_MATCHES
    emit(f"benchmark_v2_pitchcontrol  gt={a.gt}  beta={a.beta}  frames/match<={a.frames}  test={tests}")
    rng = np.random.default_rng(0)
    for nz in a.noise.split(","):
        methods = load_selected(a.tuning_dir, a.gt, nz)
        t0 = time.time()
        rows = []
        for mid in tests:
            rows += run_match(a.data, mid, a.gt, nz, methods, a.frames, a.beta)
        df = pd.DataFrame(rows, columns=["match", "frame", "method", "mae_map", "n_home", "n_away"])
        df.to_csv(os.path.join(a.out, f"pc_frames_{a.gt}_{nz}.csv"), index=False)
        pm = df.groupby(["match", "method"], as_index=False)["mae_map"].mean()
        pm.to_csv(os.path.join(a.out, f"pc_per_match_{a.gt}_{nz}.csv"), index=False)
        emit("\n" + "=" * 96)
        emit(f"NOISE {nz}  ground truth {a.gt}   ({time.time()-t0:.0f}s, frames evaluated per match: "
             f"{df.groupby('match').frame.nunique().to_dict()})")
        emit("=" * 96)
        g = pm.groupby("method")["mae_map"]
        s = pd.DataFrame({"mean_MAEmap": g.mean(), "sd_across_matches": g.std()})
        piv = pm.pivot(index="match", columns="method", values="mae_map")
        s["mean_rank"] = piv.rank(axis=1).mean()
        s["best_in_n_matches"] = (piv.rank(axis=1) == 1).sum()
        s["within_match_sd"] = df.groupby("method")["mae_map"].std()
        emit(s.sort_values("mean_MAEmap").to_string())
        emit("\nPaired per-match differences (negative = first method has LOWER distortion); "
             "95% CI from bootstrap over matches:")
        for x_, y_ in [("BW(tuned)", "MA(tuned)"), ("BW(tuned)", "STATIC(zero velocity)"),
                       ("MA(tuned)", "STATIC(zero velocity)"), ("KF(tuned)", "MA(tuned)"),
                       ("MA(paper-default)", "MA(tuned)"), ("BW(paper-default)", "BW(tuned)")]:
            r = paired(pm, x_, y_, rng)
            if r:
                emit(f"  {x_:<22} - {y_:<24} mean {r[0]:+.4f}  CI [{r[1]:+.4f}, {r[2]:+.4f}]  "
                     f"lower in {r[3]}/{r[4]} matches")
    with open(os.path.join(a.out, f"pipeline_output_v2_pitchcontrol_{a.gt}.txt"), "w") as f:
        f.write("\n".join(log))


if __name__ == "__main__":
    main()