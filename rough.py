import json

with open('/Users/hritikshukla/Documents/Reasearch Work aka project/data/archive_skillcorner/1886347/1886347_match.json') as f:
    match = json.load(f)

with open('/Users/hritikshukla/Documents/Reasearch Work aka project/data/archive_skillcorner/1886347/extracted_player_tracking.json') as f:
    tracking = json.load(f)

# What IDs does match.json give?
match_ids = [str(p['trackable_object']) for p in match['players']]
print("match.json IDs sample:", match_ids[:5])

# What IDs does tracking have?
track_ids = list(tracking.keys())
print("tracking IDs sample:", track_ids[:5])

# How many overlap?
overlap = set(match_ids) & set(track_ids)
print("overlapping IDs:", len(overlap))