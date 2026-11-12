import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from config import pitch_len, pitch_width, output_dir, match_id

os.makedirs(output_dir, exist_ok=True)

def plot_velocity_comparision(v_raw, v_filtered, v_star,
                            filter_label = "Savitzky-Golay",
                            player_id = ""):
    fig, ax = plt.subplots(figsize=(12, 4))
    frames = np.arange(len(v_raw))

    ax.plot(
        frames,
        v_star,
        color='#E74C3C',
        linewidth=2.0,
        label='Ground truth',
        zorder=3,
        )
    ax.plot(
        frames,
        v_filtered,
        color='#2E86AB',
        linewidth=1.2,
        label=filter_label,
        zorder=2,
        )
    ax.plot(
        frames,
        v_raw,
        color='#BBBBBB',
        linewidth=0.5,
        label='Raw corrupted velocity',
        zorder=1,
        )

    ax.set_ylim(0, 30)
    ax.set_xlabel('Frame number', fontsize=11)
    ax.set_ylabel('Speed (m/s)', fontsize=11)
    ax.set_title(
        f'Fig 1 — Raw vs Filtered Velocity | Player {player_id}\n'
        '(raw spikes truncated at 30 m/s for clarity; peak spike = 120 m/s)',
        fontsize=11,
    )
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "fig1_velocity_comparison.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_fig1_velocity_comparison(v_raw, v_filtered, v_star,
                                  filter_label = "Savitzky-Golay",
                                  player_id = ""):
    return plot_velocity_comparision(v_raw, v_filtered, v_star,
                                     filter_label=filter_label,
                                     player_id=player_id)
    
def plot_pitch_control(pc_home, title = "Pitch Control Surface"):
    fig, ax = plt.subplots(figsize = (12, 7))
    
    im = ax.imshow(
        pc_home.T,
        origin='lower',
        extent=[0, pitch_len, 0, pitch_width],
        cmap='RdBu_r',
        vmin=0, vmax=1,
        aspect='auto'
    )
    _draw_pitch(ax)
    plt.colorbar(im, ax=ax,
                 label='P(Home Control) — 1=Home, 0=Away, 0.5=Contested')
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Pitch Length (m)")
    ax.set_ylabel("Pitch Width (m)")

    path = os.path.join(output_dir, "fig2_pitch_control.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

def plot_fig2_pitch_control(pc_home, title = "Pitch Control Surface"):
    return plot_pitch_control(pc_home, title=title)

def plot_difference_map(PC_a, PC_b,
                         label_a="Filter A",
                         label_b="Filter B"):

    diff = PC_a - PC_b

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(
        diff.T,
        origin='lower',
        extent=[0, pitch_len, 0, pitch_width],
        cmap='coolwarm',
        vmin=-0.3, vmax=0.3,
        aspect='auto'
    )
    _draw_pitch(ax)
    plt.colorbar(im, ax=ax,
                 label=f'ΔPC: positive = {label_a} gives more home control')
    ax.set_title(f"Pitch Control Difference — {label_a} vs {label_b}",
                 fontsize=13)
    ax.set_xlabel("Pitch Length (m)")
    ax.set_ylabel("Pitch Width (m)")

    fname = f"fig3_diff_{label_a.replace(' ','_')}_vs_{label_b.replace(' ','_')}.png"
    path  = os.path.join(output_dir, fname)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def plot_pareto_scatter(rmse_scores, timing_scores):
    labels = {
        'moving_avg'     : 'Moving Average',
        'savitzky_golay' : 'Savitzky-Golay',
        'butterworth'    : 'Butterworth (1.5Hz)',
        'kalman'         : 'Kalman (CV)',
    }
    colors = {
        'moving_avg'     : 'steelblue',
        'savitzky_golay' : 'crimson',
        'butterworth'    : 'seagreen',
        'kalman'         : 'darkorange',
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    for fname, label in labels.items():
        x = timing_scores.get(fname)
        y = rmse_scores.get(fname)
        if x is None or y is None:
            continue
        ax.scatter(x, y, s=120, color=colors[fname],
                   zorder=5, label=label)
        ax.annotate(label, (x, y),
                    textcoords='offset points',
                    xytext=(8, 4), fontsize=9)

    ax.set_xlabel("Wall-clock time per match half (s)")
    ax.set_ylabel("RMSE (m/s)")
    ax.set_title("Fig 3 — Accuracy–Compute Trade-off\n"
                 "(Pareto-optimal = bottom-left)")
    ax.legend()
    ax.grid(True, ls='--', alpha=0.4)

    ax.annotate("← Faster", xy=(0.05, 0.08),
                xycoords='axes fraction', fontsize=9, color='grey')
    ax.annotate("↓ More\naccurate", xy=(0.85, 0.12),
                xycoords='axes fraction', fontsize=9, color='grey')

    path = os.path.join(output_dir, "fig4_pareto_scatter.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

def _draw_pitch(ax):
    ax.plot([0, pitch_len, pitch_len, 0, 0],
            [0, 0, pitch_width, pitch_width, 0],
            'k-', lw=1.5, zorder=3)

    ax.axvline(pitch_len / 2, color='k', lw=1,
               ls='--', alpha=0.5, zorder=3)

    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(pitch_len / 2 + 9.15 * np.cos(theta),
            pitch_width / 2 + 9.15 * np.sin(theta),
            'k-', lw=1, alpha=0.4, zorder=3)