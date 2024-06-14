import json

# Load the original JSON file
with open('/storage/mmodel-soup/sarc_data/data_split_output/sarc_AS_dataset_train.json', 'r') as f:
    data = json.load(f)

# Transform the data to a list of dictionaries
data_list = [{'id': f'{k}.jpg', 'text': v['text'], 'label': v['label']} for k, v in data.items()]

# Write the transformed data to a new JSON file
with open('/storage/mmodel-soup/sarc_data/data_split_output/sarc_AS_dataset_train_new.json', 'w') as f:
    json.dump(data_list, f, indent=4)

print("New JSON file created successfully.")


# Load the original JSON file
with open('/storage/mmodel-soup/sarc_data/data_split_output/sarc_R_dataset_train.json', 'r') as f:
    data = json.load(f)

# Transform the data to a list of dictionaries
data_list = [{'id': f'{k}.jpg', 'text': v['text'], 'label': v['label']} for k, v in data.items()]

# Write the transformed data to a new JSON file
with open('/storage/mmodel-soup/sarc_data/data_split_output/sarc_R_dataset_train_new.json', 'w') as f:
    json.dump(data_list, f, indent=4)

print("New JSON file created successfully.")


# Load the original JSON file
with open('/storage/mmodel-soup/sarc_data/data_split_output/sarc_U_dataset_train.json', 'r') as f:
    data = json.load(f)

# Transform the data to a list of dictionaries
data_list = [{'id': f'{k}.jpg', 'text': v['text'], 'label': v['label']} for k, v in data.items()]

# Write the transformed data to a new JSON file
with open('/storage/mmodel-soup/sarc_data/data_split_output/sarc_U_dataset_train_new.json', 'w') as f:
    json.dump(data_list, f, indent=4)

print("New JSON file created successfully.")



import json

import json

def jsonl_to_dict(jsonl_file):
    """
    Reads a JSONL file and returns a dictionary with image_id as keys and descriptions as values.

    Parameters:
    jsonl_file (str): Path to the JSONL file

    Returns:
    dict: A dictionary with image_id as keys and descriptions as values
    """
    image_dict = {}
    
    with open(jsonl_file, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            image_dict[data["image_id"]] = data["description"]
    
    return image_dict

def dict_to_json(output_file, data_dict):
    """
    Writes a dictionary to a JSON file.

    Parameters:
    output_file (str): Path to the output JSON file
    data_dict (dict): The dictionary to write to the JSON file
    """
    with open(output_file, 'w') as file:
        json.dump(data_dict, file, indent=4)

# Example usage
jsonl_file = '/storage/mmodel-soup/sarc_data/data_gen_output/sarc_image_description.jsonl'
output_json_file = 'sarcasm_image_descriptions_final.json'

# Convert JSONL to dictionary
image_dict = jsonl_to_dict(jsonl_file)

# Write dictionary to JSON file
dict_to_json(output_json_file, image_dict)

# Print the dictionary to verify
print(image_dict)
