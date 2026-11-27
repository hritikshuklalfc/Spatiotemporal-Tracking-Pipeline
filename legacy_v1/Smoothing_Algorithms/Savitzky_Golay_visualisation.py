import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import json 

file_path = "data/archive_skillcorner/1899585/extracted_player_tracking.json"
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

#Savitzky–Golay velocity calculation
polyorder = 2
df['X_SG'] = df.groupby('player_id')['X'].transform(lambda x: savgol_filter(x, window_size, polyorder) if len(x) > window_size else x)
df['Y_SG'] = df.groupby('player_id')['Y'].transform(lambda x: savgol_filter(x, window_size, polyorder) if len(x) > window_size else x)

dx_sg = ((df["X_SG"] - df.groupby("player_id")["X_SG"].shift(1))**2)
dy_sg = ((df["Y_SG"] - df.groupby("player_id")["Y_SG"].shift(1))**2)
df['SG_Velocity'] = np.sqrt(dx_sg + dy_sg) / dt

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

#plotting the Savitzky–Golay velocity
plt.plot(
    plot_df["Frame"], 
    plot_df["SG_Velocity"], 
    color = "#FF3366", 
    linewidth = 2, 
    label = "Savitzky-Golay (Preserves Peaks)", 
    alpha = 1.0, 
    zorder = 3
)

#adding titles and labels
plt.title(f"Savitzky Golay for Player {first_player}", color="#0d0d0d", fontsize=16)
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

