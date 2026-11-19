#!/usr/bin/env python3
"""
benchmark_v2.py -- corrected kinematic benchmark for the filter study
=====================================================================
Run from the REPOSITORY ROOT (same place you run main.py / multi_match.py from):

    python Spearman_Pitch_Control/benchmark_v2.py --gt sg15
    python Spearman_Pitch_Control/benchmark_v2.py --gt gauss     # robustness check

What this fixes compared with the v1 pipeline (all verified on the v1 code):
  1. Independent noise per player and per match (v1 re-seeded with 42 for every
     player, so all 309 players received the SAME noise sequence).
  2. Evaluation only on physically plausible frames. In v1, ~49% of the
     ">7 m/s sprint band" frames were >12.5 m/s tracking glitches in the
     SkillCorner positions (max 144.7 m/s; median run length 1 frame).
  3. Equal tuning for all filters: every filter family is tuned over a grid on
     TUNING matches and scored on separate TEST matches.
  4. Edge handling: moving average uses edge padding (v1 zero-padded), and the
     first/last TRIM frames are excluded for every filter.
  5. Robust baselines: a rolling-median filter, and a Hampel outlier pre-filter
     that can be placed in front of ANY filter (fair for all four).
  6. A second, non-i.i.d. noise model (N2: correlated noise + persistent
     identity-switch offsets + freeze-and-jump dropouts), next to the paper's
     noise (N1), so the ranking is not tested only on the family that favours
     a wide uniform average.
  7. A raw (unfiltered) baseline row, and the paper's fixed settings re-scored
     under the new protocol for direct comparison.
  8. Alternative ground truth (--gt gauss) so the result is not tied to the
     Savitzky-Golay(15,2) ground truth that equals the competing SG filter.

NOT covered here: the pitch-control (MAE_map) part. It must be re-run after the
kinematic part is settled.

Every constant in the RESEARCHER DECISIONS block is a choice that belongs to
the researcher. The N2 parameters in particular are PROPOSALS, not facts.
"""
import argparse
import json
import os
import time
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, butter, filtfilt, lfilter, ss2tf
from scipy.ndimage import median_filter, gaussian_filter1d

# =========================== RESEARCHER DECISIONS ===========================
FS = 10
DT = 1.0 / FS
MATCH_IDS = ["1886347", "1899585", "1925299", "1953632", "1996435",
             "2006229", "2011166", "2013725", "2015213", "2017461"]
TUNE_MATCHES = MATCH_IDS[:3]          # parameters are chosen on these
TEST_MATCHES = MATCH_IDS[3:]          # and scored on these (never used for tuning)
BASE_SEED = 42
MIN_LEN = 200                         # skip very short player series

V_MAX_PLAUSIBLE = 12.0                # m/s; GT speeds above this are treated as tracking glitches
GLITCH_DILATE = 10                    # frames excluded on each side of a glitch (SG window spreads it)
TRIM = 20                             # frames excluded at both ends of every series

# N1 = the paper's noise (isolated single-frame spikes, i.i.d. Gaussian)
N1 = dict(sigma=0.4, p_occ=0.05, spike=8.0)
# N2 = proposed non-i.i.d. noise
N2 = dict(sigma=0.4, rho=0.7,                       # AR(1) correlated positional noise
          p_burst=0.002, burst_len=(3, 20), burst_mag=(2.0, 8.0),   # persistent identity-switch offsets
          p_freeze=0.002, freeze_len=(5, 30))                       # position frozen, then jumps back

GT_GAUSS_SIGMA = 2.5                  # frames; alternative ground truth (--gt gauss)
HAMPEL = dict(half=3, scale_win=31, nsig=3.0, min_thr=0.5)   # metres

GRIDS = dict(
    MA=[dict(k=k) for k in (5, 7, 9, 11, 13, 15, 21, 31, 45, 61)],
    SG=[dict(w=w, p=p) for w in (7, 9, 11, 13, 15, 21, 31, 45, 61) for p in (2, 3)],
    BW=[dict(fc=fc) for fc in (0.15, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)],
    KF=[dict(sR=r, sQ=q) for r in (0.1, 0.2, 0.4, 1.0, 1.5, 3.0, 6.0) for q in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)],
    MED=[dict(k=k) for k in (3, 5, 7, 9, 15, 21, 31, 45)],
)
PAPER_DEFAULTS = {
    "MA": dict(k=15), "SG": dict(w=15, p=2), "BW": dict(fc=1.5), "KF": dict(sR=1.5, sQ=0.01),
}
# ============================================================================


# ------------------------------ filters (1-D) ------------------------------
def f_ma(z, k):
    h = k // 2
    return np.convolve(np.pad(z, h, mode="edge"), np.ones(k) / k, mode="valid")


def f_sg(z, w, p):
    return savgol_filter(z, w, p)


def f_bw(z, fc):
    b, a = butter(2, fc / (FS / 2.0), btype="low")
    return filtfilt(b, a, z)


def f_med(z, k):
    return median_filter(z, size=k, mode="nearest")


def kf_gain(sR, sQ):
    F = np.array([[1, 0, DT, 0], [0, 1, 0, DT], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
    Q = np.eye(4) * sQ ** 2
    R = np.eye(2) * sR ** 2
    P = np.eye(4)
    for _ in range(50):                      # same steady-state recipe as filters.py
        P = F @ P @ F.T + Q
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        P = (np.eye(4) - K @ H) @ P
    return K


_KF_CACHE = {}


def kf_tf(sR, sQ):
    """Steady-state CV Kalman filter as an LTI transfer function (exactly the
    recursion in filters.py, but vectorised with lfilter instead of a Python loop)."""
    key = (sR, sQ)
    if key not in _KF_CACHE:
        K = kf_gain(sR, sQ)
        k = np.array([[K[0, 0]], [K[2, 0]]])
        F2 = np.array([[1.0, DT], [0.0, 1.0]])
        H2 = np.array([[1.0, 0.0]])
        A = (np.eye(2) - k @ H2) @ F2
        C = np.array([[1.0, 0.0]])
        b, a = ss2tf(A, k, C @ A, C @ k)
        _KF_CACHE[key] = (b[0], a)
    return _KF_CACHE[key]


def f_kf(z, sR, sQ):
    b, a = kf_tf(sR, sQ)
    z0 = z[0]
    return z0 + lfilter(b, a, z - z0)       # initial state [z0, 0], as in filters.py


def hampel(z):
    r = z - median_filter(z, size=2 * HAMPEL["half"] + 1, mode="nearest")
    s = 1.4826 * median_filter(np.abs(r), size=HAMPEL["scale_win"], mode="nearest")
    thr = np.maximum(HAMPEL["nsig"] * s, HAMPEL["min_thr"])
    out = z.copy()
    bad = np.abs(r) > thr
    out[bad] = (z - r)[bad]                 # replace by the rolling median
    return out


FAMILY_FN = {
    "MA": lambda z, p: f_ma(z, p["k"]),
    "SG": lambda z, p: f_sg(z, p["w"], p["p"]),
    "BW": lambda z, p: f_bw(z, p["fc"]),
    "KF": lambda z, p: f_kf(z, p["sR"], p["sQ"]),
    "MED": lambda z, p: f_med(z, p["k"]),
}


def apply_cfg(xn, yn, family, params, pre):
    if pre == "hampel":
        xn, yn = hampel(xn), hampel(yn)
    fn = FAMILY_FN[family]
    return fn(xn, params), fn(yn, params)


# ------------------------------ ground truth -------------------------------
def gt_positions(x, y, kind):
    if kind == "sg15":
        return savgol_filter(x, 15, 2), savgol_filter(y, 15, 2)
    if kind == "gauss":
        return (gaussian_filter1d(x, GT_GAUSS_SIGMA, mode="nearest"),
                gaussian_filter1d(y, GT_GAUSS_SIGMA, mode="nearest"))
    raise ValueError(kind)


def speed(x, y):
    return np.hypot(np.diff(x), np.diff(y)) / DT


def valid_mask(v_gt):
    bad = (v_gt > V_MAX_PLAUSIBLE).astype(float)
    bad_d = np.convolve(bad, np.ones(2 * GLITCH_DILATE + 1), mode="same") > 0
    ok = ~bad_d
    ok[:TRIM] = False
    ok[-TRIM:] = False
    return ok


# --------------------------------- noise -----------------------------------
def make_rng(match_id, player_idx, tag):
    return np.random.default_rng(np.random.SeedSequence([BASE_SEED, int(match_id), player_idx, tag]))


def noise_n1(xs, ys, rng):
    n = len(xs)
    ex = rng.normal(0, N1["sigma"], n)
    ey = rng.normal(0, N1["sigma"], n)
    d = np.where(rng.uniform(size=n) < N1["p_occ"], N1["spike"], 0.0)
    return xs + ex + d, ys + ey + d


def _ar1(rng, n, sigma, rho):
    return lfilter([1.0], [1.0, -rho], rng.normal(0, sigma * np.sqrt(1 - rho ** 2), n))


def noise_n2(xs, ys, rng):
    n = len(xs)
    xn = xs + _ar1(rng, n, N2["sigma"], N2["rho"])
    yn = ys + _ar1(rng, n, N2["sigma"], N2["rho"])
    for s in np.where(rng.uniform(size=n) < N2["p_burst"])[0]:       # identity-switch-like offsets
        L = int(rng.integers(N2["burst_len"][0], N2["burst_len"][1] + 1))
        mag = rng.uniform(*N2["burst_mag"])
        ang = rng.uniform(0, 2 * np.pi)
        xn[s:s + L] += mag * np.cos(ang)
        yn[s:s + L] += mag * np.sin(ang)
    for s in np.where(rng.uniform(size=n) < N2["p_freeze"])[0]:      # frozen extrapolation, then jump
        L = int(rng.integers(N2["freeze_len"][0], N2["freeze_len"][1] + 1))
        xn[s:s + L] = xn[s]
        yn[s:s + L] = yn[s]
    return xn, yn


NOISE = {"N1": (noise_n1, 1), "N2": (noise_n2, 2)}


# --------------------------------- data ------------------------------------
def load_match(data_root, mid):
    path = os.path.join(data_root, mid, "extracted_player_tracking.json")
    with open(path) as f:
        d = json.load(f)
    players = []
    for pid, p in d.items():
        x = np.asarray(p["x"], float)
        y = np.asarray(p["y"], float)
        if len(x) < MIN_LEN:
            continue
        for z in (x, y):
            m = np.isnan(z)
            if m.any():
                z[m] = np.interp(np.flatnonzero(m), np.flatnonzero(~m), z[~m])
        players.append((pid, x, y))
    return players


# -------------------------------- scoring ----------------------------------
def score(v_true, v_f, ok):
    e = (v_true - v_f)[ok]
    a, b = v_true[ok], v_f[ok]
    r = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan
    return np.sqrt(np.mean(e ** 2)), np.mean(np.abs(e)), r


def band_masks(v_true, ok):
    a = np.gradient(v_true, DT)
    return {
        "Low(<=4)": ok & (v_true <= 4.0),
        "HSR(5.5-7)": ok & (v_true >= 5.5) & (v_true <= 7.0),
        "Sprint(7-12)": ok & (v_true > 7.0),
        "Accel(|a|>3)": ok & (np.abs(a) > 3.0),
    }


def band_mae(v_true, v_f, ok):
    return {k: (np.mean(np.abs(v_true - v_f)[m]) if m.sum() > 0 else np.nan)
            for k, m in band_masks(v_true, ok).items()}


def player_stream(data_root, matches, gt_kind, noise_name):
    """Yield (match, player, v_true, ok, xn, yn) with independent noise per player."""
    nfn, tag = NOISE[noise_name]
    for mid in matches:
        for pi, (pid, x, y) in enumerate(load_match(data_root, mid)):
            xs, ys = gt_positions(x, y, gt_kind)
            v_true = speed(xs, ys)
            ok = valid_mask(v_true)
            if ok.sum() < 100:
                continue
            xn, yn = nfn(xs.copy(), ys.copy(), make_rng(mid, pi, tag))
            yield mid, pid, v_true, ok, xn, yn


# ------------------------------- self tests --------------------------------
def selftest():
    rng = np.random.default_rng(0)
    z = np.cumsum(rng.normal(0, 0.3, 2000)) + rng.normal(0, 0.4, 2000) + 30
    # vectorised Kalman == the loop implementation used in filters.py
    sR, sQ = 1.5, 0.01
    K = kf_gain(sR, sQ)
    c, v, out = z[0], 0.0, []
    for zi in z:
        p = c + v * DT
        r = zi - p
        c = p + K[0, 0] * r
        v = v + K[2, 0] * r
        out.append(c)
    d = np.max(np.abs(np.array(out) - f_kf(z, sR, sQ)))
    assert d < 1e-8, f"vectorised Kalman differs from loop version by {d}"
    # moving average has no zero-padding edge artefact
    s = f_ma(np.full(100, 30.0), 15)
    assert np.allclose(s, 30.0), "MA edge handling wrong"
    # noise is different for different players, identical when seeded identically
    a = noise_n1(np.zeros(500), np.zeros(500), make_rng("1886347", 0, 1))[0]
    b = noise_n1(np.zeros(500), np.zeros(500), make_rng("1886347", 1, 1))[0]
    assert not np.allclose(a, b), "noise should differ between players"
    # an isolated N1 spike really is 8 m on both axes = ~113 m/s
    print("self-test OK: Kalman vectorisation max diff %.2e; N1 spike speed = %.1f m/s"
          % (d, np.hypot(N1["spike"], N1["spike"]) / DT))


# --------------------------------- tuning ----------------------------------
def all_configs():
    return [(fam, p) for fam, grid in GRIDS.items() for p in grid]


def tune(data_root, gt_kind, noise_name, objective, quick):
    """Returns {pre: {family: (best_params, table_df)}}."""
    cfgs = all_configs()
    if quick:
        cfgs = [c for c in cfgs if c[0] in ("MA", "BW") or (c[0] == "KF" and c[1]["sQ"] in (0.01, 0.1))]
    rows = []
    cnt = 0
    for mid, pid, v_true, ok, xn, yn in player_stream(data_root, TUNE_MATCHES[:1] if quick else TUNE_MATCHES,
                                                         gt_kind, noise_name):
        cnt += 1
        for pre in ("none", "hampel"):
            xh, yh = (hampel(xn), hampel(yn)) if pre == "hampel" else (xn, yn)
            for fam, p in cfgs:
                fn = FAMILY_FN[fam]
                rm, ma_, _ = score(v_true, speed(fn(xh, p), fn(yh, p)), ok)
                rows.append((pre, fam, json.dumps(p, sort_keys=True), rm, ma_))
    df = pd.DataFrame(rows, columns=["pre", "family", "params", "rmse", "mae"])
    agg = df.groupby(["pre", "family", "params"], as_index=False)[["rmse", "mae"]].mean()
    best = {}
    for pre in ("none", "hampel"):
        best[pre] = {}
        for fam in agg.family.unique():
            t = agg[(agg.pre == pre) & (agg.family == fam)].sort_values(objective).reset_index(drop=True)
            best[pre][fam] = (json.loads(t.loc[0, "params"]), t)
    return best, agg, cnt


def edge_warning(fam, params):
    grid = GRIDS[fam]
    for key in grid[0]:
        vals = sorted({g[key] for g in grid})
        if params[key] in (vals[0], vals[-1]) and len(vals) > 2:
            return f"{key}={params[key]} is at the grid edge"
    return ""


# -------------------------------- evaluation -------------------------------
def evaluate(data_root, gt_kind, noise_name, best, quick):
    """Score selected configs, paper defaults, and the raw baseline on TEST matches."""
    methods = {}   # name -> (family, params, pre)
    for pre in ("none", "hampel"):
        for fam, (p, _) in best[pre].items():
            tag = "" if pre == "none" else "Hampel+"
            methods[f"{tag}{fam}(tuned)"] = (fam, p, pre)
    for fam, p in PAPER_DEFAULTS.items():
        methods[f"{fam}(paper-default)"] = (fam, p, "none")
    per_match = []
    bands = []
    tests = TEST_MATCHES[:1] if quick else TEST_MATCHES
    for mid, pid, v_true, ok, xn, yn in player_stream(data_root, tests, gt_kind, noise_name):
        rm, ma_, r = score(v_true, speed(xn, yn), ok)
        per_match.append((mid, "RAW(unfiltered)", rm, ma_, r))
        bands.append((mid, "RAW(unfiltered)", band_mae(v_true, speed(xn, yn), ok)))
        for name, (fam, p, pre) in methods.items():
            xf, yf = apply_cfg(xn, yn, fam, p, pre)
            vf = speed(xf, yf)
            rm, ma_, r = score(v_true, vf, ok)
            per_match.append((mid, name, rm, ma_, r))
            bands.append((mid, name, band_mae(v_true, vf, ok)))
    pm = pd.DataFrame(per_match, columns=["match", "method", "rmse", "mae", "r"])
    pm = pm.groupby(["match", "method"], as_index=False).mean()
    bd = pd.DataFrame([dict(match=m, method=n, **b) for m, n, b in bands])
    bd = bd.groupby(["match", "method"], as_index=False).mean()
    return pm, bd, methods


def summarise(pm):
    g = pm.groupby("method")
    s = g[["rmse", "mae", "r"]].mean()
    s["rmse_sd_matches"] = g["rmse"].std()
    ranks = pm.pivot(index="match", columns="method", values="rmse").rank(axis=1)
    s["mean_rank"] = ranks.mean()
    s["best_in_n_matches"] = (ranks == 1).sum()
    return s.sort_values("rmse")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="sg15", choices=["sg15", "gauss"])
    ap.add_argument("--objective", default="rmse", choices=["rmse", "mae"])
    ap.add_argument("--data", default="data/archive_skillcorner")
    ap.add_argument("--out", default="outputs/v2")
    ap.add_argument("--quick", action="store_true", help="smoke test on 1 tuning + 1 test match")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    selftest()
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    log = []

    def emit(s=""):
        print(s, flush=True)
        log.append(str(s))

    emit(f"benchmark_v2  gt={a.gt}  objective={a.objective}  tune={TUNE_MATCHES}  test={TEST_MATCHES}")
    for noise_name in ("N1", "N2"):
        t0 = time.time()
        best, agg, ntune = tune(a.data, a.gt, noise_name, a.objective, a.quick)
        agg.to_csv(os.path.join(a.out, f"tuning_{a.gt}_{noise_name}.csv"), index=False)
        emit("\n" + "=" * 100)
        emit(f"NOISE {noise_name}  ground truth {a.gt}  (tuned on {ntune} players, {time.time()-t0:.0f}s)")
        emit("=" * 100)
        emit("Selected parameters (by %s on tuning matches):" % a.objective)
        for pre in ("none", "hampel"):
            for fam, (p, t) in best[pre].items():
                w = edge_warning(fam, p)
                emit(f"  {'Hampel+' if pre=='hampel' else '       '}{fam:<4} {p}  tune-{a.objective}={t.loc[0, a.objective]:.3f}"
                     + (f"   [!] {w}" if w else ""))
        pm, bd, methods = evaluate(a.data, a.gt, noise_name, best, a.quick)
        pm.to_csv(os.path.join(a.out, f"per_match_{a.gt}_{noise_name}.csv"), index=False)
        bd.to_csv(os.path.join(a.out, f"bands_{a.gt}_{noise_name}.csv"), index=False)
        emit(f"\nTEST-match results (n={pm.match.nunique()} matches; mean over players, then matches):")
        emit(summarise(pm).to_string())
        emit("\nIntensity-band MAE (m/s) on plausible frames, TEST matches:")
        bsum = bd.drop(columns="match").groupby("method").mean()
        emit(bsum.loc[summarise(pm).index].to_string())
    with open(os.path.join(a.out, f"pipeline_output_v2_{a.gt}.txt"), "w") as f:
        f.write("\n".join(log))
    emit(f"\nSaved to {a.out}/")


if __name__ == "__main__":
    main()