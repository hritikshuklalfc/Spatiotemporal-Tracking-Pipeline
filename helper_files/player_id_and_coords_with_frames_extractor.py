import json

#creating a set to store the player_id for furthur operation
unique_player_id = set()

#providing the file_path for the document
file_path = "/Users/hritikshukla/Documents/Reasearch Work aka project/data/archive_skillcorner/1886347/1886347_tracking_extrapolated.jsonl"

#reading the json file and extracting the player_id
with open(file_path, "r") as file:
    for line in file:
        data = json.loads(line)
        
        for player in data.get("player_data", []):
            player_id = player.get('player_id')
            if player_id is not None:
                unique_player_id.add(player_id)

#creating a prefilled (empty) data holder to put in the values with the player_id being the key
player_tracking_data = {
    player_id : {"x":[], "y":[], "frame":[]} for player_id in unique_player_id
}

#readig the file and appending the info into the player_tracking_data
with open(file_path, "r") as file:
    for line in file:
        data = json.loads(line)
        frame = data.get("frame")
        
        for player in data.get("player_data", []):
            player_id = player.get("player_id")
        
            #getting the value of x and y coords        
            if player_id is not None:
                x_coords = player.get("x")
                y_coords = player.get("y")
                
                player_tracking_data[player_id]["x"].append(x_coords)
                player_tracking_data[player_id]["y"].append(y_coords)
                player_tracking_data[player_id]["frame"].append(frame)

#creating the filepath for exportting the data
output_file_path = "/Users/hritikshukla/Documents/Reasearch Work aka project/data/archive_skillcorner/1886347/extracted_player_tracking.json"

#using json.dump to export the info, in the formatted way
with open(output_file_path, "w") as outfile:
    json.dump(player_tracking_data, outfile, indent=4)

#confirmation message
print(f"Congratulations, u did some shit, you file will be here: {output_file_path}")