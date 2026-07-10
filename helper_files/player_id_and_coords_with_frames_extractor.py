import json
from collections import defaultdict

MATCH_IDS = [
    "1886347",   # match 1 — already validated
    "1899585",   # match 2 — already validated
    "1925299",
    "1953632",
    "1996435",
    "2006229",
    "2011166",
    "2013725",
    "2015213",
    "2017461",
]

for i in MATCH_IDS:
    # 1. Provide the file_path for the document
    file_path = f"data/archive_skillcorner/{i}/{i}_tracking_extrapolated.jsonl"
    
    # 2. Use defaultdict to automatically build the empty lists for new player_ids
    # This REPLACES your entire first loop and the unique_player_id set!
    player_tracking_data = defaultdict(lambda: {"x": [], "y": [], "frame": []})

    # 3. Read the file ONCE and append the info directly
    with open(file_path, "r") as file:
        for line in file:
            # Clean the line of invisible newline characters/spaces
            clean_line = line.strip()
            
            # If the line is empty after cleaning, skip to the next line
            if not clean_line:
                continue
                
            try:
                data = json.loads(clean_line)
            except json.JSONDecodeError:
                print(f"Warning: Found unreadable JSON line in match {i}. Skipping it.")
                continue

            frame = data.get("frame")
            
            for player in data.get("player_data", []):
                player_id = player.get("player_id")
            
                if player_id is not None:
                    # defaultdict ensures we can just append without checking if the key exists yet
                    player_tracking_data[player_id]["x"].append(player.get("x"))
                    player_tracking_data[player_id]["y"].append(player.get("y"))
                    player_tracking_data[player_id]["frame"].append(frame)

    # 4. Create the filepath for exporting the data
    output_file_path = f"/Users/hritikshukla/Documents/Reasearch Work aka project/data/archive_skillcorner/{i}/extracted_player_tracking.json"

    # 5. Export the info (converting defaultdict back to a standard dict)
    with open(output_file_path, "w") as outfile:
        json.dump(dict(player_tracking_data), outfile, indent=4)

    # Confirmation message
    print(f"Congratulations, u did some shit, your file will be here: {output_file_path}")
    
print("All Done! No Fucking Error")