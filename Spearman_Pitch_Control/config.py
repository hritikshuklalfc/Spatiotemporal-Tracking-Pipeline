#----------Data Path-----------#
match_id = "1886347"
base_path = f"data/archive_skillcorner/{match_id}"

match_json = f"{base_path}/{match_id}_match.json"
tracking_json   = f"{base_path}/extracted_player_tracking.json"
phases_csv      = f"{base_path}/{match_id}_phases_of_play.csv"
output_dir     = "outputs"

#---------Pitch Dimensions------------#
pitch_len = 104.0
pitch_width = 68.0

#---------Team ID-----------#
home_team_id = 4177
away_team_id = 1805

#--------Tracking Sample Rate-------------#
fps = 10
dt = 1.0/fps

#---------------Synthetic Noise Perameter----------------------#
sigma = 0.4
p_occ = 0.05
spike_mag = 8.0
noise_seed = 42

#--------------Filter Perameter-------------#
ma_half_window = 3

gt_sg_window = 15
gt_sg_polyorder = 2

sg_window = 7
sg_polyorder = 3

bw_butoff_hz = 2.0
bw_order = 4
bw_fs = fps

kf_q_std = 0.1
kf_r_std = fps

#-------------Pitch Control Model--------------#
vmax = 8.0
tr = 0.7
beta = 2.5

grid_nx = 52
grid_ny = 34

#----------Evaluation-----------#
n_eval_frames = 25
eval_period = 1
min_played = 5