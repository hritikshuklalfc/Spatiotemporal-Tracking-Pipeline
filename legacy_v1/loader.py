import json
import numpy as np
import pandas as pd
from config import (match_json, tracking_json, phases_csv, 
                    home_team_id, n_eval_frames, eval_period)

def load_player_team_map():
    with open(match_json) as f:
        match = json.load(f)
        
    player_team = {}
    for p in match["players"]:
        pid = str(p["id"])
        player_team[pid] = p["team_id"]
        
    return player_team

def load_tracking():
    with open(tracking_json) as f:
        tracking = json.load(f)
        
    n_players = len(tracking)
    sample_pid = list(tracking.keys())[0]
    n_frames = len(tracking[sample_pid]["x"])
    
    return tracking
        
def load_active_frame():
    df = pd.read_csv(phases_csv)
    
    df = df[df["period"].astype(int) == eval_period].copy()
    df["frame_start"] = df["frame_start"].astype(int)
    df["frame_end"] = df["frame_end"].astype(int)
    
    df["frame_mid"] = ((df["frame_start"] + df["frame_end"])/2).astype(int)
    
    candidates = df["frame_mid"].values
    idx = np.linspace(0, len(candidates) - 1, n_eval_frames, dtype=int)
    selected = sorted(candidates[idx].tolist())
    
    return selected

def build_frame_index(tracking):
    ref_pid = list(tracking.keys())[0]
    ref_frames = tracking[ref_pid]['frame']
    return {f: i for i, f in enumerate(ref_frames)}