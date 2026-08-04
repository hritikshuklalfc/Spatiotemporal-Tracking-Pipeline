import pandas as pd
import os

match_ids = [
    "1886347",  
    "1899585",   
    "1925299",
    "1953632",
    "1996435",
    "2006229",
    "2011166",
    "2013725",
    "2015213",
    "2017461",
    ]  

for mid in match_ids:
    df = pd.read_csv(f"data/archive_skillcorner/{mid}/{mid}_phases_of_play.csv")
    
    print(f"\nMatch {mid}")
    print(f"  Total phases:          {len(df)}")
    print(f"  Avg phase duration:    {df['duration'].mean():.2f}s")
    print(f"  Avg possessions/phase: {df['n_player_possessions_in_phase'].mean():.2f}")
    print(f"  Avg team width:        {df['team_in_possession_width_start'].mean():.2f}m")