import json, numpy as np
d = json.load(open("data/archive_skillcorner/1886347/extracted_player_tracking.json"))
for pid, p in list(d.items())[:5]:
    f = np.array(p["frame"]); print(pid, len(f), "gaps:", (np.diff(f) != 1).sum(), "largest:", np.diff(f).max())