import json
import ipdb
image_description_dict = {}

jsonl_file = "/root/zqi2/mmodel-soup/sarc_data/intermediate_data/sarc_image_description_sample.jsonl"

with open(jsonl_file, "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line)
        
        image_id = data["image_id"]
        description = data["description"]
        
        # Add to the dictionary
        image_description_dict[image_id] = description

output_json_file = "sarc_image_descriptions.json"

# Save the dictionary to a JSON file
with open(output_json_file, "w") as json_file:
    json.dump(image_description_dict, json_file)

print("JSON file saved successfully.")