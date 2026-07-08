import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, butter, filtfilt
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

#butterwroth velocity calculation
#human movement usually under 1-2 Hz, lower the smoother but might cutoff the sudden sprints, so experimenting a bit with this value
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

#plotting the Butterworth velocity
plt.plot(
    plot_df["Frame"], 
    plot_df["BW_Velocity"], 
    color = "#2ca02c", 
    linewidth = 3.5, 
    label = "Butterwoth velocity calculation (Peak Preserved)", 
    alpha = 0.9, 
    zorder = 2
)


#adding titles and labels
plt.title(f"Butterworth for Player {first_player}", color="#0d0d0d", fontsize=16)
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