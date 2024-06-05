import os
import json
import ipdb

# Define the directory where the JSON files are located
directory = '/storage/mmodel-soup/nycartoon_data/data_split_output'
train_path = '/storage/mmodel-soup/nycartoon_data/data_raw/nycartoon_dataset_train.json'
with open(train_path, 'r') as file:
    # Load the JSON data
    train_data = json.load(file)

# Loop through all files in the directory
for filename in os.listdir(directory):
    # Check if the file is a JSON file
    if filename.endswith('.json'):
        # Construct the full file path
        file_path = os.path.join(directory, filename)
        
        # Open the JSON file
        with open(file_path, 'r') as file:
            # Load the JSON data
            data = json.load(file)
            for observation in data:
                data[observation]['description'] = train_data[observation]['description']
                data[observation]['mode'] = "single"
            with open(f"{file_path}_NEW.json", "w") as file:
                # Write the dictionary to the file in JSON format
                json.dump(data, file, indent=4)