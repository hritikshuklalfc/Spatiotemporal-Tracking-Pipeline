import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import json 

file_path = "data/archive_skillcorner/1886347/extracted_player_tracking.json"
all_tracking_data = []

#finding the required data
with open(file_path, "r") as file:
    player_data = json.load(file)
    
player_ids = player_data.keys()
player_ids_int = [(player_id) for player_id in player_data.keys()]

for player_id, data in player_data.items():
    frames = data.get("frame", [])
    x_coords = data.get("x", [])
    y_coords = data.get("y", [])
    
    for frame, x, y in zip(frames, x_coords, y_coords):
        all_tracking_data.append([int(player_id), frame, x, y])
    
df = pd.DataFrame(all_tracking_data, columns=["player_id", "Frame", "X", "Y"])

#cleaning and interpolation
df["X"] = df["X"].interpolate(method="linear", limit=5)
df["Y"] = df["Y"].interpolate(method="linear", limit=5)
df = df.dropna().reset_index(drop=True)

#other required data
fps = 10
dt = 1/fps
window_size = 15

#raw velocity calculation
dx = ((df["X"] - df.groupby("player_id")["X"].shift(1))**2)
dy = ((df["Y"] - df.groupby("player_id")["Y"].shift(1))**2)
df['Raw_velocity'] = np.sqrt(dx + dy) / dt

#kalman's velocity calculation
df["X_Kalman"] = np.nan
df["y_Kalman"] = np.nan
df["Vx_Kalman"] = np.nan
df["Vy_Kalman"] = np.nan

#state transition matrix F: models constant velocity (x = x + vx * dt)
F = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

#observation matrix H: we only observe X and Y (not velocity)
H = np.array([[1, 0, 0, 0], 
              [0, 1, 0, 0]])

#Measurement Noise covariance R: How much we trust the raw data (Camera Noise)
R = np.eye(2)*1.5

#process noise covarience Q: How much accelration person undergo
q_var = 4.0
Q = np.array([[dt**4/4, 0, dt**3/2, 0],
              [0, dt**4/4, 0, dt**3/2],
              [dt**3/2, 0, dt**2, 0],
              [0, dt**3/2, 0, dt**2]]) * q_var

#applying killman filter per player
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


#using just the first player for the graph
first_player = int(player_ids_int[0])
plot_df = df[df["player_id"] == first_player].copy()

#ploting the graph
plt.figure(figsize=(14, 6))

#plotting the raw velocity
plt.plot(
    plot_df["Frame"], 
    plot_df["Raw_velocity"], 
    color = "#5D5757", 
    linewidth=3, 
    label = "Raw Camera Noisy Data", 
    alpha = 0.7
)

#plotting the rolling average velocity
plt.plot(
    plot_df["Frame"], 
    plot_df["Kalman_Velocity"], 
    color = "#0B057F", 
    linewidth = 3.5, 
    label = "Kalman Filter", 
    alpha = 0.9, 
    zorder = 2
)

#adding titles and labels
plt.title(f"Kalman Filter for Player {first_player}", color="#0d0d0d", fontsize=16)
plt.xlabel("Frame Number", color="#0d0d0d", fontsize=12)
plt.ylabel("Speed (in m/sec)", color="#0d0d0d", fontsize=12)

#adding legends
plt.legend(facecolor="#ffffff", edgecolor="#070505", labelcolor="#0d0d0d")

#making it a bit asthetic
plt.gca().set_facecolor("#ffffff")
plt.gcf().patch.set_facecolor("#ffffff")
plt.tick_params(color="#224b45", labelcolor="#ffffff") 
plt.grid(color="#333333", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()