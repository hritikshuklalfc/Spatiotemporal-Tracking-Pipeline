#----------Data Path-----------#
match_ids = [
    "1886347", "1899585", "1925299", "1953632", "1996435",
    "2006229", "2011166", "2013725", "2015213", "2017461",
]
match_id = "1886347"
base_path = f"data/archive_skillcorner/{match_id}"

match_json = f"{base_path}/{match_id}_match.json"
tracking_json   = f"{base_path}/extracted_player_tracking.json"
phases_csv      = f"{base_path}/{match_id}_phases_of_play.csv"
output_dir     = f"outputs/match_{match_id}"

#---------Pitch Dimensions------------#
pitch_len = 104.0
pitch_width = 68.0

#---------Team ID-----------#
import json


def _load_team_ids():
	try:
		with open(match_json) as f:
			match = json.load(f)
		return match["home_team"]["id"], match["away_team"]["id"]
	except Exception:
		return 4177, 1805


home_team_id, away_team_id = _load_team_ids()

#--------Tracking Sample Rate-------------#
fps = 10
dt = 1.0/fps

#---------------Synthetic Noise Perameter----------------------#
sigma = 0.4
p_occ = 0.05
spike_mag = 8.0
noise_seed = 42

#--------------Filter Perameter-------------#
ma_half_window = 7

gt_sg_window = 15
gt_sg_polyorder = 2

sg_window = 15
sg_polyorder = 2

bw_butoff_hz = 1.5
bw_order = 2
bw_fs = fps

kf_q_std = 0.01
kf_r_std = 1.5

#-------------Pitch Control Model--------------#
vmax = 8.0
tr = 0.7
beta = [2.3, 2.5, 2.7]

grid_nx = 52
grid_ny = 34

#----------Evaluation-----------#
n_eval_frames = 25
eval_period = 1
min_played = 5