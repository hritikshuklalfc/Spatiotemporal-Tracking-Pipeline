import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt
from config import (dt, sigma, p_occ, spike_mag, noise_seed,
                    ma_half_window, sg_window, sg_polyorder,
                    bw_butoff_hz, bw_order, bw_fs,
                    kf_q_std, kf_r_std, gt_sg_window, gt_sg_polyorder)

def build_ground_truth(x_clean, y_clean):
    x = np.array(x_clean, dtype=float)
    y = np.array(y_clean, dtype=float)
    
    x_star = savgol_filter(x, window_length=gt_sg_window, polyorder=gt_sg_polyorder)
    y_star = savgol_filter(y, window_length=gt_sg_window, polyorder=gt_sg_polyorder)
    
    vx_star = np.diff(x_star) / dt
    vy_star = np.diff(y_star) / dt
    
    return x_star, y_star, vx_star, vy_star

def inject_noise(x_star, y_star, seed = noise_seed):
    n = len(x_star)    
    rng = np.random.default_rng(seed)
    
    eta_x = rng.normal(0, sigma, size = n)
    eta_y = rng.normal(0, sigma, size = n)
    
    occ_mask = rng.uniform(0, 1, size = n) < p_occ
    d_x = np.where(occ_mask, spike_mag, 0.0)
    d_y = np.where(occ_mask, spike_mag, 0.0)
    
    x_noise = x_star + eta_x + d_x
    y_noise = y_star + eta_y + d_y
    
    return x_noise, y_noise
        
def _velocity(xf, yf):
    return np.diff(xf) / dt, np.diff(yf) / dt


def filter_moving_average(x_noisy, y_noisy, k=ma_half_window):
    kernel = np.ones(2 * k + 1) / (2 * k + 1)
    x_ma   = np.convolve(x_noisy, kernel, mode='same')
    y_ma   = np.convolve(y_noisy, kernel, mode='same')
    return _velocity(x_ma, y_ma)

def filter_savitzky_golay(x_noisy, y_noisy,
                           window=sg_window,
                           poly=sg_polyorder):

    x_sg = savgol_filter(x_noisy, window_length=window, polyorder=poly)
    y_sg = savgol_filter(y_noisy, window_length=window, polyorder=poly)
    return _velocity(x_sg, y_sg)

def filter_butterworth(x_noisy, y_noisy,
                        fc=bw_butoff_hz,
                        fs=bw_fs,
                        order=bw_order):
    nyq  = fs / 2.0
    b, a = butter(order, fc / nyq, btype='low')

    x_bw = filtfilt(b, a, x_noisy)
    y_bw = filtfilt(b, a, y_noisy)
    return _velocity(x_bw, y_bw)

def filter_kalman(x_noisy, y_noisy,
                   q_std=kf_q_std,
                   r_std=kf_r_std):

    n = len(x_noisy)

    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1,  0],
                  [0, 0, 0,  1]], dtype=float)

    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]], dtype=float)

    Q = np.eye(4) * q_std ** 2  
    R = np.eye(2) * r_std ** 2  

    state = np.array([x_noisy[0], y_noisy[0], 0.0, 0.0])
    P     = np.eye(4)

    xs, ys = [], []

    for i in range(n):
        state = F @ state
        P     = F @ P @ F.T + Q

        z     = np.array([x_noisy[i], y_noisy[i]])
        y_res = z - H @ state                 
        S     = H @ P @ H.T + R               
        K     = P @ H.T @ np.linalg.inv(S)    
        state = state + K @ y_res
        P     = (np.eye(4) - K @ H) @ P

        xs.append(state[0])
        ys.append(state[1])

    x_kf = np.array(xs)
    y_kf = np.array(ys)
    return _velocity(x_kf, y_kf)

def compute_all_filters(x_clean, y_clean):
    x = np.array(x_clean, dtype=float)
    y = np.array(y_clean, dtype=float)
    
    if len(x) < sg_window + 2:
        return None
    
    x_star, y_star, vx_star, vy_star = build_ground_truth(x, y)
    
    x_noisy, y_noisy = inject_noise(x_star, y_star)
    
    vx_raw = np.diff(x_noisy) / dt
    vy_raw = np.diff(y_noisy) / dt
    
    return {
        'ground_truth'   : (vx_star,                          vy_star),
        'raw_noisy'      : (vx_raw,                           vy_raw),
        'moving_avg'     : filter_moving_average(x_noisy,     y_noisy),
        'savitzky_golay' : filter_savitzky_golay(x_noisy,     y_noisy),
        'butterworth'    : filter_butterworth(x_noisy,        y_noisy),
        'kalman'         : filter_kalman(x_noisy,             y_noisy),
    }