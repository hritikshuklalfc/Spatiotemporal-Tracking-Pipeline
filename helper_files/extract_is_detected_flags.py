import json
from pathlib import Path

MATCH_IDS = [
    "1886347", "1899585", "1925299", "1953632", "1996435",
    "2006229", "2011166", "2013725", "2015213", "2017461",
]

BASE_DIR = "data/archive_skillcorner"

for match_id in MATCH_IDS:
    jsonl_path = f"{BASE_DIR}/{match_id}/{match_id}_tracking_extrapolated.jsonl"
    out_path = f"{BASE_DIR}/{match_id}/detection_mask.json"
    
    if not Path(jsonl_path).exists():
        print(f"[SKIP] {match_id} -- No JSONL found")
        continue
    
    masks = {}
    
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            frame = d['frame']
            
            for p in d["player_data"]:
                pid = str(p["player_id"])
                if pid not in masks:
                    masks[pid] = {'frame': [], "is_detected": []}
                masks[pid]["frame"].append(frame)
                masks[pid]["is_detected"].append(bool(p["is_detected"]))
    
    with open(out_path, "w") as f:
        json.dump(masks, f)
        
    all_detected = []
    for pid, data in masks.items():
        all_detected.extend(data["is_detected"])
        
    det_rate = sum(all_detected) / len(all_detected) * 100
    print(f"Match {match_id}: {len(masks)} players"
          f"{det_rate:.1f}% frames detected, "
          f"{100-det_rate:.1f}% extrapolated")