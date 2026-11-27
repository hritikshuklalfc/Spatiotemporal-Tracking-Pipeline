#!/usr/bin/env python3
"""
make_figures_v2.py -- regenerate every figure of the revised paper
==================================================================
Run from the REPOSITORY ROOT after benchmark_v2.py and benchmark_v2_pitchcontrol.py
have produced their CSV files in outputs/v2/:

    python Spearman_Pitch_Control/make_figures_v2.py --gt sg15

Figures (all saved as PNG at 300 dpi and PDF into outputs/v2/figures/):
  fig1_gaps          the data pitfall: frame-number gaps create fake speeds
  fig2_tuning        velocity error vs smoothing parameter (paper defaults marked)
  fig3_velocity      velocity error per method, tuned vs paper defaults
  fig4_pitchcontrol  pitch-control distortion per method, with the "ignore velocity" line
  fig5_surfaces      one real frame: ground-truth surface and error maps

The old Fig. 1-4 must NOT be reused: they were produced from the flawed series.
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

import benchmark_v2 as B
import benchmark_v2_pitchcontrol as P

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 9, "figure.dpi": 120})
BLUE, ORANGE, GREY, RED = "#1f77b4", "#ff7f0e", "#7f7f7f", "#d62728"


from matplotlib.ticker import FuncFormatter, NullFormatter


def plain_log(ax, axis="x", ticks=None):
    a = ax.xaxis if axis == "x" else ax.yaxis
    if ticks is not None:
        (ax.set_xticks if axis == "x" else ax.set_yticks)(ticks)
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_minor_formatter(NullFormatter())


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ----------------------------------------------------------------- Figure 1
def fig1(data, out, mid="1886347"):
    with open(os.path.join(data, mid, "extracted_player_tracking.json")) as f:
        d = json.load(f)
    gaps = []
    for p in d.values():
        st = np.diff(np.asarray(p["frame"]))
        gaps += list(st[st > 1])
    gaps = np.array(gaps)
    # pick a player with several gaps in a 9000-frame window
    best = None
    for pid, p in d.items():
        fr = np.asarray(p["frame"])
        if len(fr) < 20000:
            continue
        w0 = fr[len(fr) // 3]
        m = (fr >= w0) & (fr < w0 + 9000)
        ng = int((np.diff(fr[m]) > 1).sum())
        if best is None or ng > best[0]:
            best = (ng, pid, w0)
    _, pid, w0 = best
    p = d[pid]
    fr = np.asarray(p["frame"])
    x, y = np.asarray(p["x"], float), np.asarray(p["y"], float)
    sel = (fr >= w0 - 20) & (fr < w0 + 9000)
    fr, x, y = fr[sel], x[sel], y[sel]
    # v1 treatment: rows treated as consecutive
    xs, ys = savgol_filter(x, 15, 2), savgol_filter(y, 15, 2)
    v1 = np.hypot(np.diff(xs), np.diff(ys)) / B.DT
    gap_at = fr[1:][np.diff(fr) > 1]
    # corrected treatment: split at gaps
    cuts = np.where(np.diff(fr) != 1)[0] + 1
    xf, vf = [], []
    for idx in np.split(np.arange(len(fr)), cuts):
        if len(idx) < 40:
            continue
        a, b = savgol_filter(x[idx], 15, 2), savgol_filter(y[idx], 15, 2)
        v = np.hypot(np.diff(a), np.diff(b)) / B.DT
        xf += list(fr[idx][1:]) + [np.nan]
        vf += list(v) + [np.nan]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.1), gridspec_kw={"width_ratios": [1, 1.5, 1.5]})
    ax[0].hist(gaps, bins=np.logspace(0, np.log10(gaps.max() + 1), 30), color=GREY)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("size of gap in frame numbers (frames)")
    ax[0].set_ylabel("number of gaps")
    ax[0].set_title(f"(a) gaps in all players, match {mid}\nmedian {int(np.median(gaps))} frames")
    ax[1].plot(fr[1:], v1, lw=0.6, color=RED)
    for g in gap_at:
        ax[1].axvline(g, color="k", lw=0.4, alpha=0.5)
    ax[1].set_ylim(0, 40)
    ax[1].set_xlabel("frame number")
    ax[1].set_ylabel("speed (m/s)")
    ax[1].set_title(f"(b) v1: rows treated as consecutive\n(peak {v1.max():.0f} m/s; black lines = gaps)")
    ax[2].plot(xf, vf, lw=0.6, color=BLUE)
    for g in gap_at:
        ax[2].axvline(g, color="k", lw=0.4, alpha=0.5)
    ax[2].set_ylim(0, 40)
    ax[2].set_xlabel("frame number")
    ax[2].set_title("(c) corrected: series split at gaps")
    fig.tight_layout()
    save(fig, out, "fig1_gaps")


# ----------------------------------------------------------------- Figure 2
def fig2(res, gt, out):
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.2))
    spec = [("BW", "fc", "cutoff frequency fc (Hz)", 1.5), ("MA", "k", "window k (frames)", 15),
            ("SG", "w", "window w (frames)", 15)]
    for r, nz in enumerate(("N1", "N2")):
        t = pd.read_csv(os.path.join(res, f"tuning_{gt}_{nz}.csv"))
        for c, (fam, key, lab, dflt) in enumerate(spec):
            ax = axes[r, c]
            for pre, ls in (("none", "-"), ("hampel", "--")):
                s = t[(t.pre == pre) & (t.family == fam)].copy()
                s["params"] = s.params.apply(json.loads)
                s["x"] = s.params.apply(lambda q: q[key])
                if fam == "SG":
                    s = s[s.params.apply(lambda q: q["p"] == 3)]
                s = s.sort_values("x")
                ax.plot(s.x, s.rmse, ls, marker="o", ms=3, color=BLUE if pre == "none" else ORANGE,
                        label="filter" if pre == "none" else "Hampel + filter")
                i = s.rmse.idxmin()
                ax.plot(s.loc[i, "x"], s.loc[i, "rmse"], "*", ms=11, color="k", zorder=5)
            ax.axvline(dflt, color=RED, ls=":", lw=1)
            ax.set_xscale("log")
            ax.set_yscale("log")
            plain_log(ax, "x", {"BW": [0.1, 0.25, 0.5, 1, 2], "MA": [5, 15, 30, 60, 120], "SG": [7, 15, 30, 60, 120]}[fam])
            plain_log(ax, "y", [0.3, 0.5, 1, 2, 5, 10])
            ax.set_xlabel(lab)
            if c == 0:
                ax.set_ylabel(f"{nz}: velocity RMSE (m/s)")
            ax.set_title({"BW": "Butterworth", "MA": "Moving average", "SG": "Savitzky-Golay (order 3)"}[fam])
            if r == 0 and c == 0:
                ax.legend(frameon=False, fontsize=7)
    fig.text(0.5, -0.01, "star = best on tuning matches; red dotted line = setting used in the first version of the paper",
             ha="center", fontsize=8)
    fig.tight_layout()
    save(fig, out, "fig2_tuning")


# ------------------------------------------------------- Figures 3 and 4
ORDER = ["Hampel+BW(tuned)", "BW(tuned)", "Hampel+SG(tuned)", "SG(tuned)", "Hampel+MA(tuned)", "MA(tuned)",
         "Hampel+KF(tuned)", "KF(tuned)", "MA(paper-default)", "SG(paper-default)", "KF(paper-default)",
         "BW(paper-default)", "STATIC(zero velocity)", "RAW(unfiltered)"]


def dotplot(res, gt, fname, valcol, xlabel, title, out, name, static_line=False):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, nz in zip(axes, ("N1", "N2")):
        d = pd.read_csv(os.path.join(res, fname.format(gt=gt, nz=nz)))
        names = [m for m in ORDER if m in set(d.method)]
        for yi, m in enumerate(names):
            v = d[d.method == m][valcol].values
            col = ORANGE if "paper-default" in m else (GREY if ("STATIC" in m or "RAW" in m) else BLUE)
            ax.scatter(v, np.full(len(v), yi), s=12, color=col, alpha=0.6)
            ax.plot([v.mean()], [yi], "|", color="k", ms=12)
        if static_line and "STATIC(zero velocity)" in names:
            ax.axvline(d[d.method == "STATIC(zero velocity)"][valcol].mean(), color=GREY, ls="--", lw=1)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_ylim(len(names) - 0.5, -0.5)          # best (first in ORDER) at the top, set once
        ax.set_xscale("log")
        plain_log(ax, "x", [0.3, 0.5, 1, 2, 5, 10, 30] if valcol == "rmse" else [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1])
        ax.set_xlabel(xlabel)
        ax.set_title(f"{nz}: {title}")
    fig.text(0.5, -0.02, "dots = the 7 test matches, black tick = mean; blue = tuned, orange = first-version defaults, grey = baselines",
             ha="center", fontsize=8)
    fig.tight_layout()
    save(fig, out, name)


# ----------------------------------------------------------------- Figure 5
def fig5(res, gt, data, out, noise="N1", mid=None):
    from pitch_control import pitch_control_frame, compute_mae_map
    mid = mid or B.TEST_MATCHES[0]
    methods = P.load_selected(res, gt, noise)
    keep = {k: methods[k] for k in ("MA(paper-default)", "BW(tuned)", "Hampel+BW(tuned)") if k in methods}
    with open(os.path.join(data, mid, f"{mid}_match.json")) as f:
        match = json.load(f)
    team_of = {str(p["id"]): p["team_id"] for p in match["players"]}
    team_of["home"], team_of["away"] = match["home_team"]["id"], match["away_team"]["id"]
    segs = B.load_match(data, mid)
    frames = P.select_frames(segs, team_of, 60)
    f = frames[len(frames) // 2]
    pcf = os.path.join(res, f"pc_frames_{gt}_{noise}.csv")
    if os.path.exists(pcf):                 # representative frame: closest to each method's own mean
        df = pd.read_csv(pcf)
        df = df[df.match.astype(str) == str(mid)]
        piv = df.pivot(index="frame", columns="method", values="mae_map")
        cols = [c for c in ("STATIC(zero velocity)", "MA(paper-default)", "BW(tuned)") if c in piv]
        if cols and len(piv):
            z = ((piv[cols] - piv[cols].mean()) / piv[cols].mean()).abs().sum(axis=1)
            f = int(z.idxmin())
    nfn, tag = B.NOISE[noise]
    players = []
    for si_all, (pid, si, f0, x, y) in enumerate(segs):
        team = team_of.get(str(pid))
        n = len(x)
        if team not in (team_of["home"], team_of["away"]) or not (f0 + B.TRIM + 1 <= f <= f0 + n - 1 - B.TRIM):
            continue
        xs, ys = B.gt_positions(x, y, gt)
        ok = B.valid_mask(B.speed(xs, ys))
        i = int(f - f0)
        if not ok[i - 1]:
            continue
        xn, yn = nfn(xs.copy(), ys.copy(), B.make_rng(mid, si_all, tag))
        vel = {"GT": (np.diff(xs)[i - 1] / B.DT, np.diff(ys)[i - 1] / B.DT), "STATIC": (0.0, 0.0)}
        for name, (fam, p, pre) in keep.items():
            xf, yf = B.apply_cfg(xn, yn, fam, p, pre)
            vel[name] = (np.diff(xf)[i - 1] / B.DT, np.diff(yf)[i - 1] / B.DT)
        players.append(dict(team=team, px=float(x[i]) + P.OFF_X, py=float(y[i]) + P.OFF_Y, vel=vel))

    def pc(key):
        h = [dict(px=q["px"], py=q["py"], vx=q["vel"][key][0], vy=q["vel"][key][1]) for q in players if q["team"] == team_of["home"]]
        a = [dict(px=q["px"], py=q["py"], vx=q["vel"][key][0], vy=q["vel"][key][1]) for q in players if q["team"] == team_of["away"]]
        return pitch_control_frame(h, a, 2.5)
    star = pc("GT")
    panels = [("STATIC", "ignore velocity"), ("MA(paper-default)", "moving average, first-version k=15"),
              ("BW(tuned)", "Butterworth, tuned")]
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6), layout="constrained")
    ext = [0, 104, 0, 68]
    im = ax[0].imshow(star.T, origin="lower", extent=ext, cmap="RdBu_r", vmin=0, vmax=1, aspect="equal")
    for q in players:
        c = "#8b0000" if q["team"] == team_of["home"] else "#00008b"
        ax[0].plot(q["px"], q["py"], "o", color=c, ms=4, mec="w", mew=0.5)
        ax[0].arrow(q["px"], q["py"], q["vel"]["GT"][0] * 0.7, q["vel"]["GT"][1] * 0.7, color=c, width=0.2, head_width=1.2)
    ax[0].set_title(f"(a) reference surface\nmatch {mid}, frame {f}")
    errs = []
    for a_, (key, lab) in zip(ax[1:], panels):
        errs.append((a_, np.abs(pc(key) - star), lab))
    vmax = max(e.max() for _, e, _ in errs)
    for k, (a_, e, lab) in enumerate(errs):
        im2 = a_.imshow(e.T, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=vmax, aspect="equal")
        a_.set_title(f"({'bcd'[k]}) |error|: {lab}\nMAE_map = {e.mean():.4f}")
    fig.colorbar(im, ax=ax[0], fraction=0.05, label="P(home control)")
    fig.colorbar(im2, ax=ax[1:], fraction=0.015, label="|PC - PC_reference|")
    for a_ in ax:
        a_.set_xlabel("x (m)")
    ax[0].set_ylabel("y (m)")
    save(fig, out, "fig5_surfaces")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="sg15", choices=["sg15", "gauss"])
    ap.add_argument("--res", default="outputs/v2")
    ap.add_argument("--data", default="data/archive_skillcorner")
    ap.add_argument("--out", default="outputs/v2/figures")
    a = ap.parse_args()
    fig1(a.data, a.out)
    fig2(a.res, a.gt, a.out)
    dotplot(a.res, a.gt, "per_match_{gt}_{nz}.csv", "rmse", "velocity RMSE (m/s), log scale",
            "velocity error", a.out, "fig3_velocity")
    dotplot(a.res, a.gt, "pc_per_match_{gt}_{nz}.csv", "mae_map", "MAE_map, log scale",
            "pitch-control distortion", a.out, "fig4_pitchcontrol", static_line=True)
    fig5(a.res, a.gt, a.data, a.out)


if __name__ == "__main__":
    main()