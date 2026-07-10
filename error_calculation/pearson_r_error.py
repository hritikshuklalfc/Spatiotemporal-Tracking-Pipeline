import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.stats import pearsonr
import json 
import time

file_path = "data/archive_skillcorner/1899585/extracted_player_tracking.json"
all_tracking_data = []

# Load the JSON
with open(file_path, "r") as file:
    player_data = json.load(file)
    
first_player_id = list(player_data.keys())[0]
data = player_data[first_player_id]

frames = data.get("frame", [])
x_coords = data.get("x", [])
y_coords = data.get("y", [])

for frame, x, y in zip(frames, x_coords, y_coords):
    all_tracking_data.append([int(first_player_id), frame, x, y])

df = pd.DataFrame(all_tracking_data, columns=["player_id", "Frame", "X", "Y"])

# Cleaning and interpolation
df["X"] = df["X"].interpolate(method="linear", limit=5)
df["Y"] = df["Y"].interpolate(method="linear", limit=5)
df = df.dropna().reset_index(drop=True)

fps = 10
dt = 1/fps
window_size = 7
polyorder = 3

# 3. Raw velocity calculation 
dx = ((df["X"] - df["X"].shift(1))**2)
dy = ((df["Y"] - df["Y"].shift(1))**2)
df['Raw_velocity'] = np.sqrt(dx + dy) / dt

#------------------[ALGORITHMS]----------------------#

#|---------------------------------------------|
#|1. moving average velocity calculation       |
#|---------------------------------------------|

df["X_SMA"] = df.groupby("player_id")["X"].transform(lambda x: x.rolling(window=window_size, center=True).mean())
df["Y_SMA"] = df.groupby("player_id")["Y"].transform(lambda x: x.rolling(window=window_size, center=True).mean())

dx_sma = ((df["X_SMA"] - df.groupby("player_id")["X_SMA"].shift(1))**2)
dy_sma = ((df["Y_SMA"] - df.groupby("player_id")["Y_SMA"].shift(1))**2)
df["SMA_velocity"] = np.sqrt(dx_sma + dy_sma) / dt


#|--------------------------------|
#| 2. Savitzky–Golay calculation  |
#|--------------------------------|

if len(df) > window_size:
    df['X_SG'] = savgol_filter(df['X'], window_size, polyorder)
    df['Y_SG'] = savgol_filter(df['Y'], window_size, polyorder)
else:
    df['X_SG'] = df['X']
    df['Y_SG'] = df['Y']

dx_sg = ((df["X_SG"] - df["X_SG"].shift(1))**2)
dy_sg = ((df["Y_SG"] - df["Y_SG"].shift(1))**2)
df['orignal_vel'] = np.sqrt(dx_sg + dy_sg) / dt


#|---------------------------------------------|
#|3. butterworth average velocity calculation  |
#|---------------------------------------------|

cutoff_freq = 1.5 
order = 2 #2nd order, butterworth filter

def apply_butterworth(data, cutoff, fs, order):
    b, a = butter(order, cutoff, fs=fs, btype = "low")
    
    padlen = 3 * max(len(a), len(b))
    if len(data)>padlen:
        return filtfilt(b,a, data)
    else:
        return data
#filtering X and Y coordinates
df["X_BW"] = df.groupby("player_id")["X"].transform(lambda x: apply_butterworth(x, cutoff_freq, fps, order))
df["Y_BW"] = df.groupby("player_id")["Y"].transform(lambda x: apply_butterworth(x, cutoff_freq, fps, order))

#calculating the bw velocity
dx_bw = ((df["X_BW"]-df.groupby("player_id")["X_BW"].shift(1))**2)
dy_bw = ((df["Y_BW"]-df.groupby("player_id")["Y_BW"].shift(1))**2)

df["BW_Velocity"] = np.sqrt(dx_bw + dy_bw) / dt

#|---------------------------------|
#|4. Kalman velocity calculation   |
#|---------------------------------|

df["X_Kalman"] = np.nan
df["y_Kalman"] = np.nan
df["Vx_Kalman"] = np.nan
df["Vy_Kalman"] = np.nan

F = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

H = np.array([[1, 0, 0, 0], 
              [0, 1, 0, 0]])

R = np.eye(2)*1.5

q_var = 4.0
Q = np.array([[dt**4/4, 0, dt**3/2, 0],
              [0, dt**4/4, 0, dt**3/2],
              [dt**3/2, 0, dt**2, 0],
              [0, dt**3/2, 0, dt**2]]) * q_var

for player_id, group in df.groupby("player_id"):
    measurements = group[["X", "Y"]].values
    states = np.zeros((len(measurements), 4))
    
    x = np.array([measurements[0,0], measurements[0, 1], 0, 0])
    
    P = np.eye(4) * 10
    
    for i in range(len(measurements)):
        x = np.dot(F, x)
        P = np.dot(F, np.dot(P, F.T)) + Q
        Z = measurements[i]
        
        if not np.isnan(Z).any():
            y = Z - np.dot(H, x)
            S = np.dot(H, np.dot(P, H.T)) + R
            K = np.dot(P, np.dot(H.T, np.linalg.inv(S)))
            
            x = x + np.dot(K, y)
            P = P - np.dot(K, np.dot(H, P))
            
        states[i] = x
        
    df.loc[group.index, ["X_Kalman", "Y_Kalman", "Vx_Kalman", "Vy_Kalman"]] = states
    
df["Kalman_Velocity"] = np.sqrt(df["Vx_Kalman"]**2 + df["Vy_Kalman"]**2)

#--------------------------------------------------#

#noise injection

x_star = df["X_SG"].values
y_star = df["Y_SG"].values

v_star = np.sqrt(
    np.diff(x_star)**2 + np.diff(y_star)**2
)/dt

sigma = 0.4
p_occ = 0.05
spike_mag = 8.0
np.random.seed(42)

n = len(x_star)

eta_x = np.random.normal(0, sigma, size = n)
eta_y = np.random.normal(0, sigma, size = n)

occ_mask = np.random.uniform(0, 1, size = n) < p_occ
d_x = np.where(occ_mask, spike_mag, 0.0)
d_y = np.where(occ_mask, spike_mag, 0.0)

x_noise = x_star + eta_x + d_x
y_noise = y_star + eta_y + d_y

v_raw = np.sqrt(
    np.diff(x_noise)**2 + np.diff(y_noise)**2
)/dt

# =========================================================
# THE BENCHMARK TEST: RUNNING ALL 4 ALGORITHMS ON THE NOISE
# =========================================================

frames_in_half = 27000
players_on_pitch = 22
total_half_coords = frames_in_half * players_on_pitch
current_coords = len(x_noise)
extrapolate = total_half_coords / current_coords

#1. moving average
t0 = time.perf_counter()
x_sma_test = pd.Series(x_noise).rolling(window=15, center=True, min_periods=1).mean().values
y_sma_test = pd.Series(y_noise).rolling(window=15, center=True, min_periods=1).mean().values
v_sma_test = np.sqrt(
    np.diff(x_sma_test)**2 + np.diff(y_sma_test)**2
) / dt

t_sma = (time.perf_counter() - t0) * extrapolate

#2. Savitzky-Golay 
t0 = time.perf_counter()
x_sg_test = savgol_filter(x_noise, window_length=15, polyorder=2)
y_sg_test = savgol_filter(y_noise, window_length=15, polyorder=2)
v_sg_test = np.sqrt(
    np.diff(x_sg_test)**2 + np.diff(y_sg_test)**2
) / dt

t_sg = (time.perf_counter() - t0) * extrapolate

#3. Butterworth 
t0 = time.perf_counter()
b, a  = butter(2, 1.5, fs = fps, btype="low")
x_bw_test = filtfilt(b, a, x_noise)
y_bw_test = filtfilt(b, a, y_noise)
v_bw_test = np.sqrt(
    np.diff(x_bw_test)**2 + np.diff(y_bw_test)**2
) / dt

t_bw = (time.perf_counter() - t0) * extrapolate

#4. Kalman Filter
t0 = time.perf_counter()
states = np.zeros((len(x_noise), 4))
x_k = np.array([x_noise[0], y_noise[0], 0, 0])
P = np.eye(4) * 10

for i in range(len(x_noise)):
    x_k = np.dot(F, x_k)
    P = np.dot(F, np.dot(P, F.T)) + Q
    Z = np.array([x_noise[i], y_noise[i]])
    
    y_res = Z - np.dot(H, x_k)
    S = np.dot(H, np.dot(P, H.T)) + R
    K = np.dot(P, np.dot(H.T, np.linalg.inv(S)))
    
    x_k = x_k + np.dot(K, y_res)
    P = P - np.dot(K, np.dot(H, P))
    states[i] = x_k
    
v_kalman_test = np.sqrt(states[1:, 2]**2 + states[1:, 3]**2)
t_kalman = (time.perf_counter() - t0) * extrapolate

# =========================================================
# CALCULATE ALL METRICS FOR TABLE I
# =========================================================

def calculate_pearson(truth, filtered):
    return pearsonr(truth, filtered)[0]

test_results = {
    "Raw Corrupted Data": (v_raw, 0.00),
    "Moving Average": (v_sma_test, t_sma),
    "Savitzky-Golay": (v_sg_test, t_sg),
    "Butterworth (4th)": (v_bw_test, t_bw),
    "Kalman Filter": (v_kalman_test, t_kalman)
}

print(f"{"Algorith":<22} | {"Pearson R":<12} | {"Compute Time (sec)": <12}")
print("-" * 60)

for name, (v_filtered, compute_time) in test_results.items():
    pearson = calculate_pearson(v_star, v_filtered)
    
    print(f"{name:<22} | {pearson:<12.4f} | {compute_time:<12.4f}")
    
print("=" * 60 + "\n")